import os
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
from sklearn.metrics import accuracy_score, f1_score
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
    """Computes accuracy and F1 score for the Trainer."""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions)
    return {"accuracy": acc, "f1": f1}

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
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    
    # Remove text column as it's not needed by the PyTorch model
    tokenized_datasets = tokenized_datasets.remove_columns(["text"])
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
    
    training_args = TrainingArguments(
        output_dir=model_info['save_dir'],
        learning_rate=2e-4,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,      # 3 epochs is standard for fine-tuning
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        logging_dir=f"{model_info['save_dir']}/logs",
        logging_steps=100,
        report_to="none" # Disable wandb/tensorboard for clean terminal output
    )
    
    trainer = Trainer(
        model=peft_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    # ---------------------------------------------------------
    # 4. Train & Save
    # ---------------------------------------------------------
    print(f"Training {model_info['name']}...")
    trainer.train()
    
    print(f"Saving final {model_info['name']} LoRA adapter...")
    peft_model.save_pretrained(model_info['save_dir'])
    tokenizer.save_pretrained(model_info['save_dir'])
    print(f"Done! {model_info['name']} LoRA adapter saved to {model_info['save_dir']}")


def main():
    print("Loading primary dataset...")
    dataset = load_dataset('ErfanMoosaviMonazzah/fake-news-detection-dataset-English', cache_dir=DATASET_DIR)
    
    # To save time during dev/testing, one might want to use a small subset of the dataset
    # e.g., dataset['train'] = dataset['train'].select(range(1000))
    
    for model_info in MODELS_TO_TRAIN:
        train_lora_model(model_info, dataset)

if __name__ == "__main__":
    main()
