import os
import torch
import pickle
import numpy as np
from datasets import load_dataset
from sklearn.metrics import accuracy_score

# We can reuse the probability functions from week 3
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE_DIR, "src"))
from week3_ensemble import get_sbert_probs, get_lora_probs
from sentence_transformers import SentenceTransformer

# ==========================================
# WEEK 4: Cross-Domain Testing
# ==========================================
# PROJECT MODEL MAP:
# - Evaluates Model 1: SBERT + LogReg (from week1_baseline.py)
# - Evaluates Model 2: RoBERTa + LoRA (from week2_lora.py)
# - Evaluates Model 3: DeBERTa + LoRA (from week2_lora.py)
# - Evaluates Model 4: Stacked Ensemble 2+3 (from week3_ensemble.py)
# - Evaluates Model 5: Full Stack 1+2+3 (from week3_ensemble.py)
# ==========================================
# This script tests our 4 models on a completely different dataset (LIAR)
# to evaluate their generalization capabilities.

DATASET_DIR = os.path.join(BASE_DIR, "dataset")
SBERT_MODEL_PATH = os.path.join(BASE_DIR, "models", "sbert_baseline", "logreg_baseline.pkl")
ROBERTA_DIR = os.path.join(BASE_DIR, "models", "roberta_lora")
DEBERTA_DIR = os.path.join(BASE_DIR, "models", "deberta_lora")
ENSEMBLE_2_3_PATH = os.path.join(BASE_DIR, "models", "ensemble", "meta_learner_2_3.pkl")
ENSEMBLE_FULL_PATH = os.path.join(BASE_DIR, "models", "ensemble", "meta_learner_full.pkl")

def prepare_gonzalo_dataset():
    """Load and format the local GonzaloA dataset for cross-domain testing."""
    print("Loading local GonzaloA dataset from CSV...")
    local_csv_path = os.path.join(DATASET_DIR, "GonzaloA_FakeNews", "test.csv")
    
    # GonzaloA test.csv uses semicolon (;) as delimiter
    dataset = load_dataset('csv', data_files={"test": local_csv_path}, delimiter=';')
    
    # GonzaloA columns: [index, title, text, label]
    # We use 'text' for predictions and 'label' for truth
    texts = dataset['test']['text']
    labels = dataset['test']['label']
    
    # Filter out None/NaN text entries that would crash the models
    clean_texts = []
    clean_labels = []
    for t, l in zip(texts, labels):
        if t is not None and isinstance(t, str) and len(t.strip()) > 0 and l is not None:
            clean_texts.append(t)
            clean_labels.append(int(l))
    
    print(f"  Loaded {len(clean_texts)} valid samples (filtered {len(texts) - len(clean_texts)} invalid)")
    return clean_texts, clean_labels

def main():
    texts, true_labels = prepare_gonzalo_dataset()
    
    print(f"\nEvaluating on {len(texts)} samples from the GonzaloA dataset.")
    
    # ---------------------------------------------------------
    # 1. Evaluate SBERT Baseline
    # ---------------------------------------------------------
    print("\nLoading SBERT Baseline...")
    with open(SBERT_MODEL_PATH, 'rb') as f:
        logreg_sbert = pickle.load(f)
    model_sbert = SentenceTransformer('all-MiniLM-L6-v2')
    
    sbert_probs = get_sbert_probs(texts, model_sbert, logreg_sbert)
    sbert_preds = (sbert_probs >= 0.5).astype(int)
    sbert_acc = accuracy_score(true_labels, sbert_preds)
    print(f"SBERT Baseline Cross-Domain Accuracy: {sbert_acc * 100:.2f}%")
    
    # ---------------------------------------------------------
    # 2. Evaluate RoBERTa LoRA
    # ---------------------------------------------------------
    print("\nLoading RoBERTa LoRA...")
    roberta_probs = get_lora_probs(texts, ROBERTA_DIR, "roberta-base")
    roberta_preds = (roberta_probs >= 0.5).astype(int)
    roberta_acc = accuracy_score(true_labels, roberta_preds)
    print(f"RoBERTa LoRA Cross-Domain Accuracy: {roberta_acc * 100:.2f}%")
    
    # ---------------------------------------------------------
    # 3. Evaluate DeBERTa LoRA
    # ---------------------------------------------------------
    print("\nLoading DeBERTa LoRA...")
    deberta_probs = get_lora_probs(texts, DEBERTA_DIR, "microsoft/deberta-base")
    deberta_preds = (deberta_probs >= 0.5).astype(int)
    deberta_acc = accuracy_score(true_labels, deberta_preds)
    print(f"DeBERTa LoRA Cross-Domain Accuracy: {deberta_acc * 100:.2f}%")
    
    # ---------------------------------------------------------
    # 4. Evaluate Model 4: Stacked Ensemble (2+3)
    # ---------------------------------------------------------
    print("\nLoading Model 4: Stacked Ensemble...")
    with open(ENSEMBLE_2_3_PATH, 'rb') as f:
        meta_clf_2_3 = pickle.load(f)
        
    X_meta_2_3 = np.column_stack((roberta_probs, deberta_probs))
    ensemble_2_3_preds = meta_clf_2_3.predict(X_meta_2_3)
    ensemble_2_3_acc = accuracy_score(true_labels, ensemble_2_3_preds)
    print(f"Model 4 (Stacked 2+3) Cross-Domain Accuracy: {ensemble_2_3_acc * 100:.2f}%")
    
    # ---------------------------------------------------------
    # 5. Evaluate Model 5: Full Stack (1+2+3)
    # ---------------------------------------------------------
    print("\nLoading Model 5: Full Stack Ensemble...")
    with open(ENSEMBLE_FULL_PATH, 'rb') as f:
        meta_clf_full = pickle.load(f)
        
    X_meta_full = np.column_stack((sbert_probs, roberta_probs, deberta_probs))
    ensemble_full_preds = meta_clf_full.predict(X_meta_full)
    ensemble_full_acc = accuracy_score(true_labels, ensemble_full_preds)
    print(f"Model 5 (Full Stack) Cross-Domain Accuracy: {ensemble_full_acc * 100:.2f}%")
    
    # ---------------------------------------------------------
    # Summary Table
    # ---------------------------------------------------------
    print("\n" + "="*50)
    print("CROSS-DOMAIN GENERALIZATION RESULTS (GONZALOA DATASET)")
    print("="*50)
    print(f"Model                  | Cross-Domain Acc")
    print(f"-----------------------------------------")
    print(f"1. SBERT + LogReg      | {sbert_acc * 100:.2f}%")
    print(f"2. RoBERTa + LoRA      | {roberta_acc * 100:.2f}%")
    print(f"3. DeBERTa + LoRA      | {deberta_acc * 100:.2f}%")
    print(f"4. Stacked (2+3)       | {ensemble_2_3_acc * 100:.2f}%")
    print(f"5. Full Stack (1+2+3)  | {ensemble_full_acc * 100:.2f}%")
    print("="*50)
    print("KEY INSIGHT: Look for the model that drops the least compared to in-domain accuracy!")

if __name__ == "__main__":
    main()
