import os
import torch
import pickle
import json
import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# ==========================================
# WEEK 3: Stacked Ensemble Meta-Learner
# ==========================================
# PROJECT MODEL MAP:
# - Model 1: SBERT + LogReg (week1_baseline.py)
# - Model 2: RoBERTa + LoRA (week2_lora.py)
# - Model 3: DeBERTa + LoRA (week2_lora.py)
# - Model 4: Stacked Ensemble 2+3 (THIS FILE)
# - Model 5: Full Stack 1+2+3 (THIS FILE)
# ==========================================
# This script creates an ensemble that combines the predictions of:
# 1. Baseline SBERT + LogReg
# 2. RoBERTa + LoRA
# 3. DeBERTa + LoRA
# 4. Stacked Ensemble (Model 2 + Model 3)
# 5. Full Stack Ensemble (Model 1 + Model 2 + Model 3)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")

# Paths for the saved models
SBERT_MODEL_PATH = os.path.join(BASE_DIR, "models", "sbert_baseline", "logreg_baseline.pkl")
ROBERTA_DIR = os.path.join(BASE_DIR, "models", "roberta_lora")
DEBERTA_DIR = os.path.join(BASE_DIR, "models", "deberta_lora")
ENSEMBLE_DIR = os.path.join(BASE_DIR, "models", "ensemble")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(ENSEMBLE_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

def get_sbert_probs(texts, model_sbert, logreg_model):
    """Generate probability predictions from the SBERT Baseline."""
    embeddings = model_sbert.encode(texts, show_progress_bar=False)
    return logreg_model.predict_proba(embeddings)[:, 1] # Probability of 'Real'

def get_lora_probs(texts, model_dir, base_hf_path):
    """Generate probability predictions from a LoRA model."""
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    
    # Load base model, then apply the LoRA adapter
    base_model = AutoModelForSequenceClassification.from_pretrained(base_hf_path, num_labels=2)
    model = PeftModel.from_pretrained(base_model, model_dir)
    model.eval()
    
    # We will use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    probs = []
    # Process in batches to avoid out-of-memory
    batch_size = 16
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        inputs = tokenizer(batch_texts, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            # Apply softmax to get probabilities
            batch_probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[:, 1].cpu().numpy()
            probs.extend(batch_probs)
            
    return np.array(probs)

def main():
    print("Starting Week 3: Ensemble Building")
    
    # ---------------------------------------------------------
    # 1. Load Data from local TSV files
    # ---------------------------------------------------------
    print(f"Loading local primary dataset from TSV files...")
    local_data_dir = os.path.join(DATASET_DIR, "ErfanMoosaviMonazzah___fake-news-detection-dataset-english")
    
    data_files = {
        "train": os.path.join(local_data_dir, "train.tsv"),
        "validation": os.path.join(local_data_dir, "validation.tsv"),
        "test": os.path.join(local_data_dir, "test.tsv")
    }
    
    dataset = load_dataset('csv', data_files=data_files, delimiter='\t')
    # Use validation set to train the meta-learner to avoid overfitting on the training set
    # If no validation set exists, we split the training set, but we assume 'test' acts as our validation here.
    # In a real scenario, you'd split train into train/val. Here we use 'test' to train the ensemble,
    # or a subset of train. Let's use a subset of the training set.
    
    print("Using a 5000-sample slice of the training set to train the meta-learner...")
    meta_train_texts = dataset['train']['text'][:5000]
    meta_train_labels = dataset['train']['label'][:5000]
    
    test_split = "validation" if "validation" in dataset.keys() else "test"
    meta_test_texts = dataset[test_split]['text']
    meta_test_labels = dataset[test_split]['label']
    
    # ---------------------------------------------------------
    # 2. Generate Base Model Predictions (Features for Meta-Learner)
    # ---------------------------------------------------------
    print("\nLoading SBERT Baseline...")
    with open(SBERT_MODEL_PATH, 'rb') as f:
        logreg_sbert = pickle.load(f)
    model_sbert = SentenceTransformer('all-MiniLM-L6-v2')
    
    print("Generating SBERT predictions...")
    train_probs_sbert = get_sbert_probs(meta_train_texts, model_sbert, logreg_sbert)
    test_probs_sbert = get_sbert_probs(meta_test_texts, model_sbert, logreg_sbert)
    
    print("\nLoading RoBERTa LoRA...")
    print("Generating RoBERTa predictions...")
    train_probs_roberta = get_lora_probs(meta_train_texts, ROBERTA_DIR, "roberta-base")
    test_probs_roberta = get_lora_probs(meta_test_texts, ROBERTA_DIR, "roberta-base")
    
    print("\nLoading DeBERTa LoRA...")
    print("Generating DeBERTa predictions...")
    train_probs_deberta = get_lora_probs(meta_train_texts, DEBERTA_DIR, "microsoft/deberta-v3-base")
    test_probs_deberta = get_lora_probs(meta_test_texts, DEBERTA_DIR, "microsoft/deberta-v3-base")
    
    # ---------------------------------------------------------
    # 3. Stack Predictions into Features
    # ---------------------------------------------------------
    # Model 4 Features (RoBERTa + DeBERTa only)
    X_meta_train_2_3 = np.column_stack((train_probs_roberta, train_probs_deberta))
    X_meta_test_2_3 = np.column_stack((test_probs_roberta, test_probs_deberta))
    
    # Model 5 Features (SBERT + RoBERTa + DeBERTa)
    X_meta_train_full = np.column_stack((train_probs_sbert, train_probs_roberta, train_probs_deberta))
    X_meta_test_full = np.column_stack((test_probs_sbert, test_probs_roberta, test_probs_deberta))
    
    # ---------------------------------------------------------
    # 4. Train Model 4: Stacked Ensemble (2+3)
    # ---------------------------------------------------------
    print("\nTraining Model 4: Stacked Ensemble (RoBERTa + DeBERTa)...")
    meta_clf_2_3 = LogisticRegression()
    meta_clf_2_3.fit(X_meta_train_2_3, meta_train_labels)
    
    preds_2_3 = meta_clf_2_3.predict(X_meta_test_2_3)
    acc_2_3 = accuracy_score(meta_test_labels, preds_2_3)
    f1_2_3 = f1_score(meta_test_labels, preds_2_3)
    
    print(f"Model 4 Ensemble Accuracy: {acc_2_3 * 100:.2f}%")
    print(f"Model 4 Ensemble F1 Score: {f1_2_3 * 100:.2f}%")
    
    # ---------------------------------------------------------
    # 5. Train Model 5: Full Stack Ensemble (1+2+3)
    # ---------------------------------------------------------
    print("\nTraining Model 5: Full Stack Ensemble (SBERT + RoBERTa + DeBERTa)...")
    meta_clf_full = LogisticRegression()
    meta_clf_full.fit(X_meta_train_full, meta_train_labels)
    
    preds_full = meta_clf_full.predict(X_meta_test_full)
    acc_full = accuracy_score(meta_test_labels, preds_full)
    f1_full = f1_score(meta_test_labels, preds_full)
    
    print(f"Model 5 Full Stack Accuracy: {acc_full * 100:.2f}%")
    print(f"Model 5 Full Stack F1 Score: {f1_full * 100:.2f}%")
    
    print(f"\nModel Weights in Full Stack:")
    print(f"SBERT Weight:   {meta_clf_full.coef_[0][0]:.4f}")
    print(f"RoBERTa Weight: {meta_clf_full.coef_[0][1]:.4f}")
    print(f"DeBERTa Weight: {meta_clf_full.coef_[0][2]:.4f}")
    
    # ---------------------------------------------------------
    # 6. Save Ensembles
    # ---------------------------------------------------------
    path_2_3 = os.path.join(ENSEMBLE_DIR, "meta_learner_2_3.pkl")
    path_full = os.path.join(ENSEMBLE_DIR, "meta_learner_full.pkl")
    
    with open(path_2_3, 'wb') as f:
        pickle.dump(meta_clf_2_3, f)
    with open(path_full, 'wb') as f:
        pickle.dump(meta_clf_full, f)
        
    print(f"\nModel 4 saved to {path_2_3}")
    print(f"Model 5 saved to {path_full}")
    
    # ---------------------------------------------------------
    # 7. Save Results to results/ folder
    # ---------------------------------------------------------
    prec_2_3 = precision_score(meta_test_labels, preds_2_3)
    rec_2_3 = recall_score(meta_test_labels, preds_2_3)
    prec_full = precision_score(meta_test_labels, preds_full)
    rec_full = recall_score(meta_test_labels, preds_full)
    
    results_4 = {
        "model": "Stacked Ensemble (RoBERTa + DeBERTa)",
        "accuracy": round(acc_2_3 * 100, 2),
        "f1": round(f1_2_3 * 100, 2),
        "precision": round(prec_2_3 * 100, 2),
        "recall": round(rec_2_3 * 100, 2),
    }
    results_5 = {
        "model": "Full Stack (SBERT + RoBERTa + DeBERTa)",
        "accuracy": round(acc_full * 100, 2),
        "f1": round(f1_full * 100, 2),
        "precision": round(prec_full * 100, 2),
        "recall": round(rec_full * 100, 2),
        "meta_weights": {
            "sbert": round(meta_clf_full.coef_[0][0], 4),
            "roberta": round(meta_clf_full.coef_[0][1], 4),
            "deberta": round(meta_clf_full.coef_[0][2], 4),
        }
    }
    
    for name, data in [("week3_ensemble_2_3", results_4), ("week3_ensemble_full", results_5)]:
        path = os.path.join(RESULTS_DIR, f"{name}.json")
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"\u2713 Results saved to {path}")

if __name__ == "__main__":
    main()
