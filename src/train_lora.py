"""
LoRA fine-tuning for RoBERTa and DeBERTa.
CENG 454 — Group 2

Usage:
    python train_lora.py --model roberta
    python train_lora.py --model deberta
    python train_lora.py --model roberta --dataset sentiment140
"""
import argparse
import time
import json
import os
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import (
    AutoModelForSequenceClassification, TrainingArguments, Trainer
)
from peft import LoraConfig, TaskType, get_peft_model
from data_utils import (
    set_seed, load_imdb, load_sentiment140, tokenize_dataset,
    get_device, RESULTS_DIR, CHECKPOINTS_DIR
)

# ── Config ────────────────────────────────────────────────────
SEED = 42
MAX_LENGTH = 256
EPOCHS = 3
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.1

MODEL_CONFIGS = {
    "roberta": {
        "name": "roberta-base",
        "target_modules": ["query", "value"],
    },
    "deberta": {
        "name": "microsoft/deberta-v3-base",
        "target_modules": ["query_proj", "value_proj"],
    },
}


def compute_metrics(eval_pred):
    preds = np.argmax(eval_pred.predictions, axis=-1)
    labels = eval_pred.label_ids
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="binary"),
        "precision": precision_score(labels, preds, average="binary"),
        "recall": recall_score(labels, preds, average="binary"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["roberta", "deberta"], required=True)
    parser.add_argument("--dataset", choices=["imdb", "sentiment140"], default="imdb")
    args = parser.parse_args()

    set_seed(SEED)
    device = get_device()

    cfg = MODEL_CONFIGS[args.model]
    model_name = cfg["name"]
    tag = f"{args.model}_lora_{args.dataset}"
    output_dir = os.path.join(RESULTS_DIR, tag)
    save_dir = os.path.join(CHECKPOINTS_DIR, tag)

    # ── Data ──────────────────────────────────────────────────
    if args.dataset == "imdb":
        ds = load_imdb(ensemble_val_size=5000, seed=SEED)
    else:
        ds = load_sentiment140(subsample=50000, seed=SEED)

    tok_ds, tokenizer = tokenize_dataset(ds, model_name, MAX_LENGTH)

    # ── Model + LoRA ──────────────────────────────────────────
    base_model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2
    )

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=cfg["target_modules"],
        bias="none",
    )

    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # ── Training ──────────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=32,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        fp16=(torch.cuda.is_available() and args.model != "deberta"),
        logging_steps=100,
        seed=SEED,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tok_ds["train"],
        eval_dataset=tok_ds["test"],
        compute_metrics=compute_metrics,
    )

    print(f"\n=== Training LoRA {args.model.upper()} on {args.dataset.upper()} ===")
    start_time = time.time()
    trainer.train()
    train_time = time.time() - start_time
    print(f"Training time: {train_time:.1f}s ({train_time/60:.1f} min)")

    # ── Evaluate ──────────────────────────────────────────────
    results = trainer.evaluate()
    results["training_time_seconds"] = train_time
    results["total_params"] = total_params
    results["trainable_params"] = trainable_params
    results["lora_r"] = LORA_R
    results["lora_alpha"] = LORA_ALPHA

    print(f"\n=== LoRA {args.model.upper()} Results ===")
    for k, v in results.items():
        print(f"  {k}: {v}")

    # ── Save ──────────────────────────────────────────────────
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_dir}/metrics.json")
    print(f"Adapter saved to {save_dir}")


if __name__ == "__main__":
    main()
