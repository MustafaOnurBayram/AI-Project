import os
import time
import numpy as np
import torch
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# ==========================================
# WEEK 1: Baseline Fake News Detection
# ==========================================
# PROJECT MODEL MAP:
# - Model 1: SBERT + LogReg (THIS FILE)
# - Model 2: RoBERTa + LoRA (week2_lora.py)
# - Model 3: DeBERTa + LoRA (week2_lora.py)
# - Model 4: Stacked Ensemble 2+3 (week3_ensemble.py)
# - Model 5: Full Stack 1+2+3 (week3_ensemble.py)
# ==========================================
# This script loads the primary fake news dataset, extracts static 
# sentence embeddings using SBERT, trains a Logistic Regression classifier 
# on those embeddings, and evaluates the performance.

# Setup directory paths to store models and datasets as requested
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
MODEL_DIR = os.path.join(BASE_DIR, "models", "sbert_baseline")

# Ensure the output directories exist
os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

def main():
    print("Starting Week 1: SBERT + Logistic Regression Baseline")
    
    # ---------------------------------------------------------
    # 1. Load the primary dataset from local TSV files
    # ---------------------------------------------------------
    print(f"Loading local dataset from TSV files...")
    
    # Path to the local ErfanMoosavi dataset folder
    local_data_dir = os.path.join(DATASET_DIR, "ErfanMoosaviMonazzah___fake-news-detection-dataset-english")
    
    data_files = {
        "train": os.path.join(local_data_dir, "train.tsv"),
        "validation": os.path.join(local_data_dir, "validation.tsv"),
        "test": os.path.join(local_data_dir, "test.tsv")
    }
    
    # Load using 'csv' builder with tab delimiter
    dataset = load_dataset('csv', data_files=data_files, delimiter='\t')
    
    print("Dataset loaded successfully!")
    print(dataset)
    
    # ---------------------------------------------------------
    # 2. Extract Text and Labels
    # ---------------------------------------------------------
    # The dataset has text in 'text' column and labels in 'label' column.
    # 0 = Fake, 1 = Real
    # We will use a smaller subset for demonstration if the full dataset takes too long,
    # but by default, we'll process the full dataset for the baseline.
    
    train_texts = dataset['train']['text']
    train_labels = dataset['train']['label']
    
    # Assuming 'test' split exists. If it's 'validation', replace below.
    # We will evaluate on the test split.
    test_split_name = 'test' if 'test' in dataset.keys() else 'validation'
    test_texts = dataset[test_split_name]['text']
    test_labels = dataset[test_split_name]['label']
    
    # ---------------------------------------------------------
    # 3. Generate Static Sentence Embeddings (SBERT)
    # ---------------------------------------------------------
    # SBERT (Sentence-BERT) generates fixed-size vector representations for text.
    print("\nLoading SentenceTransformer model 'all-MiniLM-L6-v2' (fast & efficient)...")
    
    # Check for GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    model_sbert = SentenceTransformer('all-MiniLM-L6-v2', device=device)
    
    print(f"Encoding {len(train_texts)} training texts... (This may take a few minutes)")
    start_time = time.time()
    # model.encode() converts list of strings to a numpy array of shape (N, 384)
    X_train = model_sbert.encode(train_texts, show_progress_bar=True, device=device)
    print(f"Encoding training set took {time.time() - start_time:.2f} seconds.")
    
    print(f"Encoding {len(test_texts)} testing texts...")
    start_time = time.time()
    X_test = model_sbert.encode(test_texts, show_progress_bar=True, device=device)
    print(f"Encoding testing set took {time.time() - start_time:.2f} seconds.")
    
    # ---------------------------------------------------------
    # 4. Train the Baseline Classifier (Logistic Regression)
    # ---------------------------------------------------------
    print("\nTraining Logistic Regression classifier on SBERT embeddings...")
    clf = LogisticRegression(max_iter=1000)
    
    start_time = time.time()
    clf.fit(X_train, train_labels)
    print(f"Training took {time.time() - start_time:.2f} seconds.")
    
    # ---------------------------------------------------------
    # 5. Evaluate the Baseline Model
    # ---------------------------------------------------------
    print("\nEvaluating model on the test set...")
    predictions = clf.predict(X_test)
    
    accuracy = accuracy_score(test_labels, predictions)
    f1 = f1_score(test_labels, predictions)
    
    print("-" * 30)
    print(f"Baseline Expected Accuracy: ~77%")
    print(f"Actual Accuracy: {accuracy * 100:.2f}%")
    print(f"Actual F1 Score: {f1 * 100:.2f}%")
    print("-" * 30)
    
    # Generate Confusion Matrix
    cm = confusion_matrix(test_labels, predictions)
    
    # Plot Confusion Matrix using Seaborn
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Fake', 'Real'], yticklabels=['Fake', 'Real'])
    plt.title('Baseline SBERT + LogReg Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    # Save the plot in the model directory
    plot_path = os.path.join(MODEL_DIR, "confusion_matrix.png")
    plt.savefig(plot_path)
    plt.show()
    print(f"Confusion matrix plot saved to {plot_path}")
    
    # ---------------------------------------------------------
    # 6. Save the Trained Model
    # ---------------------------------------------------------
    model_path = os.path.join(MODEL_DIR, "logreg_baseline.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump(clf, f)
    print(f"Trained Logistic Regression model saved to {model_path}")

if __name__ == "__main__":
    main()
