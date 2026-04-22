"""
Baseline: Full fine-tune RoBERTa on IMDB.
CENG 454 — Group 2
"""
import time
import json
import os
import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import (
    AutoModelForSequenceClassification, TrainingArguments, Trainer
)
from datasets import concatenate_datasets
from data_utils import set_seed, load_imdb, tokenize_dataset, get_device, RESULTS_DIR, CHECKPOINTS_DIR

# ── Config ────────────────────────────────────────────────────
SEED = 42
MODEL_NAME = "roberta-base"
EPOCHS = 3
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
MAX_LENGTH = 256
OUTPUT_DIR = os.path.join(RESULTS_DIR, "roberta_full_ft")
SAVE_DIR = os.path.join(CHECKPOINTS_DIR, "roberta_full_ft")


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
    set_seed(SEED)
    device = get_device()

    # ── Data ──────────────────────────────────────────────────
    imdb = load_imdb(ensemble_val_size=5000, seed=SEED)

    # Baseline trains on FULL 25k (train + val_ensemble combined)
    full_train = concatenate_datasets([imdb["train"], imdb["val_ensemble"]])
    baseline_ds = {"train": full_train, "test": imdb["test"]}

    tok_ds, tokenizer = tokenize_dataset(baseline_ds, MODEL_NAME, MAX_LENGTH)

    # ── Model ─────────────────────────────────────────────────
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2
    )
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_params:,}")
    print(f"Trainable params: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")

    # ── Training ──────────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=32,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        fp16=torch.cuda.is_available(),
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

    print("\n=== Training Full Fine-Tune RoBERTa ===")
    start_time = time.time()
    trainer.train()
    train_time = time.time() - start_time
    print(f"Training time: {train_time:.1f}s ({train_time/60:.1f} min)")

    # ── Evaluate ──────────────────────────────────────────────
    results = trainer.evaluate()
    results["training_time_seconds"] = train_time
    results["total_params"] = total_params
    results["trainable_params"] = trainable_params

    print("\n=== Baseline Results ===")
    for k, v in results.items():
        print(f"  {k}: {v}")

    # ── Save ──────────────────────────────────────────────────
    os.makedirs(SAVE_DIR, exist_ok=True)
    trainer.save_model(SAVE_DIR)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {OUTPUT_DIR}/metrics.json")
    print(f"Model saved to {SAVE_DIR}")


if __name__ == "__main__":
    main()
