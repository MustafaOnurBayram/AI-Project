import os
import time
import json
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from peft import get_peft_model, LoraConfig, TaskType
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import numpy as np

# ==========================================
# WEEK 2: LoRA Fine-Tuning (RoBERTa & DeBERTa)
# ==========================================
# PROJECT MODEL MAP:
# - Model 1: SBERT + LogReg (week1_baseline.py)
# - Model 2: RoBERTa + LoRA (THIS FILE)
# - Model 3: DeBERTa + LoRA (THIS FILE)
# - Model 4: Stacked Ensemble 2+3 (week3_ensemble.py)
# - Model 5: Full Stack 1+2+3 (week3_ensemble.py)
# ==========================================
# This script fine-tunes RoBERTa and DeBERTa base models using LoRA
# (Low-Rank Adaptation) on the fake news dataset.

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# We will iterate over two models as requested in the PDF
MODELS_TO_TRAIN = [
    {
        "name": "RoBERTa",
        "hf_path": "roberta-base",
        "save_dir": os.path.join(BASE_DIR, "models", "roberta_lora")
    },
    {
        "name": "DeBERTa",
        "hf_path": "microsoft/deberta-base",
        "save_dir": os.path.join(BASE_DIR, "models", "deberta_lora")
    }
]

def compute_metrics(eval_pred):
    """Computes accuracy, F1, precision and recall for the Trainer."""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions)
    prec = precision_score(labels, predictions)
    rec = recall_score(labels, predictions)
    return {"accuracy": acc, "f1": f1, "precision": prec, "recall": rec}

def train_lora_model(model_info, dataset):
    print(f"\n{'='*50}")
    print(f"Starting LoRA Fine-Tuning for {model_info['name']}")
    print(f"{'='*50}")
    
    os.makedirs(model_info['save_dir'], exist_ok=True)
    
    # ---------------------------------------------------------
    # 1. Load Tokenizer & Tokenize Dataset
    # ---------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(model_info['hf_path'])
    
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

    print("Tokenizing datasets...")
    original_columns = dataset["train"].column_names
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    
    # Remove all non-essential columns to prevent PyTorch collation errors
    cols_to_remove = [col for col in original_columns if col != "label"]
    tokenized_datasets = tokenized_datasets.remove_columns(cols_to_remove)
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    tokenized_datasets.set_format("torch")
    
    train_dataset = tokenized_datasets["train"]
    # If there is no 'validation' split, fallback to 'test'
    eval_split = "validation" if "validation" in tokenized_datasets.keys() else "test"
    eval_dataset = tokenized_datasets[eval_split]

    # ---------------------------------------------------------
    # 2. Load Base Model & Apply LoRA
    # ---------------------------------------------------------
    print("Loading base model...")
    # num_labels=2 for Fake (0) vs Real (1)
    base_model = AutoModelForSequenceClassification.from_pretrained(
        model_info['hf_path'], num_labels=2
    )
    
    # Define LoRA Configuration
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS, 
        r=8,                     # Rank of the update matrices
        lora_alpha=32,           # Scaling factor
        lora_dropout=0.1,        # Dropout probability for LoRA layers
        target_modules=["query", "value"] if "deberta" not in model_info['hf_path'].lower() else ["query_proj", "value_proj"] 
    )
    
    # Wrap base model with PEFT (LoRA)
    peft_model = get_peft_model(base_model, lora_config)
    peft_model.print_trainable_parameters()
    
    # ---------------------------------------------------------
    # 3. Setup Trainer
    # ---------------------------------------------------------
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # DeBERTa crashes with FP16 gradient scaling — disable for DeBERTa only
    use_fp16 = torch.cuda.is_available() and ("deberta" not in model_info['hf_path'].lower())

    training_args = TrainingArguments(
        output_dir=model_info['save_dir'],
        learning_rate=2e-4,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        fp16=use_fp16,
        logging_steps=100,
        seed=42,
        report_to="none",
    )
    
    trainer = Trainer(
        model=peft_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    # ---------------------------------------------------------
    # 4. Train & Save
    # ---------------------------------------------------------
    print(f"Training {model_info['name']}...")
    start_time = time.time()
    trainer.train()
    train_time = time.time() - start_time
    print(f"Training time: {train_time:.1f}s ({train_time/60:.1f} min)")
    
    print(f"Saving final {model_info['name']} LoRA adapter...")
    peft_model.save_pretrained(model_info['save_dir'])
    tokenizer.save_pretrained(model_info['save_dir'])
    print(f"Done! {model_info['name']} LoRA adapter saved to {model_info['save_dir']}")
    
    # ---------------------------------------------------------
    # 5. Evaluate on test set and save results
    # ---------------------------------------------------------
    eval_split = "test" if "test" in tokenized_datasets.keys() else "validation"
    test_dataset = tokenized_datasets[eval_split]
    
    eval_results = trainer.evaluate(test_dataset)
    
    total_params = sum(p.numel() for p in peft_model.parameters())
    trainable_params = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    
    results = {
        "model": f"{model_info['name']} + LoRA",
        "accuracy": round(eval_results.get('eval_accuracy', 0) * 100, 2),
        "f1": round(eval_results.get('eval_f1', 0) * 100, 2),
        "precision": round(eval_results.get('eval_precision', 0) * 100, 2),
        "recall": round(eval_results.get('eval_recall', 0) * 100, 2),
        "total_params": total_params,
        "trainable_params": trainable_params,
        "trainable_pct": round(100 * trainable_params / total_params, 4),
        "training_time_seconds": round(train_time, 1),
        "lora_r": 8,
        "lora_alpha": 32,
        "fp16_used": use_fp16,
    }
    
    tag = model_info['name'].lower()
    results_path = os.path.join(RESULTS_DIR, f"week2_{tag}_lora.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results saved to {results_path}")
    print(f"  Accuracy: {results['accuracy']}% | F1: {results['f1']}%")
    print(f"  Trainable params: {trainable_params:,} / {total_params:,} ({results['trainable_pct']}%)")


def main():
    print("Loading local primary dataset from TSV files...")
    local_data_dir = os.path.join(DATASET_DIR, "ErfanMoosaviMonazzah___fake-news-detection-dataset-english")
    
    data_files = {
        "train": os.path.join(local_data_dir, "train.tsv"),
        "validation": os.path.join(local_data_dir, "validation.tsv"),
        "test": os.path.join(local_data_dir, "test.tsv")
    }
    
    dataset = load_dataset('csv', data_files=data_files, delimiter='\t')
    
    # To save time during dev/testing, one might want to use a small subset of the dataset
    # e.g., dataset['train'] = dataset['train'].select(range(1000))
    
    for model_info in MODELS_TO_TRAIN:
        train_lora_model(model_info, dataset)

if __name__ == "__main__":
    main()
