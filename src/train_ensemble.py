"""
Stacking ensemble: combine LoRA-RoBERTa + LoRA-DeBERTa predictions
via a Logistic Regression meta-learner.
CENG 454 — Group 2

Usage:
    python train_ensemble.py --dataset imdb
    python train_ensemble.py --dataset sentiment140
"""
import argparse
import json
import os
import pickle
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import AutoModelForSequenceClassification
from peft import PeftModel
from data_utils import (
    set_seed, load_imdb, load_sentiment140, tokenize_dataset,
    get_device, RESULTS_DIR, CHECKPOINTS_DIR
)


SEED = 42
MAX_LENGTH = 256
INFERENCE_BATCH_SIZE = 32


def get_probabilities(model, dataset, device, batch_size=32):
    """Run inference and return softmax probabilities.
    Uses manual batching to avoid DataLoader collation issues with HF datasets.
    """
    model.eval()
    model.to(device)
    all_probs = []
    all_labels = []

    n = len(dataset)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch = dataset[start:end]

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"]

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()

        all_probs.append(probs)
        if isinstance(labels, torch.Tensor):
            all_labels.extend(labels.numpy())
        else:
            all_labels.extend(labels)

        if (start // batch_size) % 50 == 0:
            print(f"  Processed {end}/{n} samples...")

    return np.vstack(all_probs), np.array(all_labels)


def load_lora_model(base_model_name, adapter_path, device):
    """Load a base model with saved LoRA adapter."""
    print(f"  Base: {base_model_name}")
    print(f"  Adapter: {adapter_path}")
    base = AutoModelForSequenceClassification.from_pretrained(
        base_model_name, num_labels=2
    )
    model = PeftModel.from_pretrained(base, adapter_path)
    model.to(device)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["imdb", "sentiment140"], default="imdb")
    args = parser.parse_args()

    set_seed(SEED)
    device = get_device()

    tag = args.dataset
    output_dir = os.path.join(RESULTS_DIR, f"ensemble_{tag}")
    os.makedirs(output_dir, exist_ok=True)

    # ── Data ──────────────────────────────────────────────────
    if args.dataset == "imdb":
        ds = load_imdb(ensemble_val_size=5000, seed=SEED)
    else:
        ds = load_sentiment140(subsample=50000, seed=SEED)

    # ── Load LoRA models ──────────────────────────────────────
    roberta_adapter = os.path.join(CHECKPOINTS_DIR, f"roberta_lora_{tag}")
    deberta_adapter = os.path.join(CHECKPOINTS_DIR, f"deberta_lora_{tag}")

    # Check adapters exist
    for path, name in [(roberta_adapter, "RoBERTa"), (deberta_adapter, "DeBERTa")]:
        if not os.path.isdir(path):
            print(f"ERROR: {name} adapter not found at {path}")
            print("Run train_lora.py first for both models.")
            return

    print("\nLoading LoRA-RoBERTa...")
    roberta_model = load_lora_model("roberta-base", roberta_adapter, device)

    print("\nLoading LoRA-DeBERTa...")
    deberta_model = load_lora_model(
        "microsoft/deberta-v3-base", deberta_adapter, device
    )

    # ── Tokenize for each model ───────────────────────────────
    print("\nTokenizing for RoBERTa...")
    rob_tok_ds, _ = tokenize_dataset(ds, "roberta-base", MAX_LENGTH)

    print("Tokenizing for DeBERTa...")
    deb_tok_ds, _ = tokenize_dataset(ds, "microsoft/deberta-v3-base", MAX_LENGTH)

    # ── Get predictions on val_ensemble (for meta-learner training) ──
    print("\n--- Generating val_ensemble predictions ---")
    rob_val_probs, val_labels = get_probabilities(
        roberta_model, rob_tok_ds["val_ensemble"], device, INFERENCE_BATCH_SIZE
    )
    deb_val_probs, _ = get_probabilities(
        deberta_model, deb_tok_ds["val_ensemble"], device, INFERENCE_BATCH_SIZE
    )

    # Stack: [rob_p_neg, rob_p_pos, deb_p_neg, deb_p_pos] → 4 features
    X_val = np.hstack([rob_val_probs, deb_val_probs])
    y_val = val_labels

    # ── Train meta-learner ────────────────────────────────────
    print(f"\nTraining meta-learner on {len(y_val)} samples...")
    meta_learner = LogisticRegression(random_state=SEED, max_iter=1000)
    meta_learner.fit(X_val, y_val)

    # Save meta-learner
    with open(os.path.join(output_dir, "meta_learner.pkl"), "wb") as f:
        pickle.dump(meta_learner, f)

    # ── Get predictions on test set ───────────────────────────
    print("\n--- Generating test predictions ---")
    rob_test_probs, test_labels = get_probabilities(
        roberta_model, rob_tok_ds["test"], device, INFERENCE_BATCH_SIZE
    )
    deb_test_probs, _ = get_probabilities(
        deberta_model, deb_tok_ds["test"], device, INFERENCE_BATCH_SIZE
    )

    X_test = np.hstack([rob_test_probs, deb_test_probs])
    y_test = test_labels

    # ── Ensemble predictions ──────────────────────────────────
    ensemble_preds = meta_learner.predict(X_test)

    results = {
        "accuracy": float(accuracy_score(y_test, ensemble_preds)),
        "f1": float(f1_score(y_test, ensemble_preds, average="binary")),
        "precision": float(precision_score(y_test, ensemble_preds, average="binary")),
        "recall": float(recall_score(y_test, ensemble_preds, average="binary")),
    }

    print(f"\n=== Stacking Ensemble Results ({tag.upper()}) ===")
    for k, v in results.items():
        print(f"  {k}: {v:.4f}")

    # ── Disagreement analysis ─────────────────────────────────
    rob_preds = np.argmax(rob_test_probs, axis=-1)
    deb_preds = np.argmax(deb_test_probs, axis=-1)

    agree = (rob_preds == deb_preds).sum()
    disagree = len(rob_preds) - agree
    both_correct = ((rob_preds == y_test) & (deb_preds == y_test)).sum()
    both_wrong = ((rob_preds != y_test) & (deb_preds != y_test)).sum()
    rob_only = ((rob_preds == y_test) & (deb_preds != y_test)).sum()
    deb_only = ((rob_preds != y_test) & (deb_preds == y_test)).sum()

    disagreement = {
        "total_samples": int(len(y_test)),
        "agree": int(agree),
        "disagree": int(disagree),
        "both_correct": int(both_correct),
        "both_wrong": int(both_wrong),
        "roberta_only_correct": int(rob_only),
        "deberta_only_correct": int(deb_only),
    }
    results["disagreement_analysis"] = disagreement

    print("\n=== Disagreement Analysis ===")
    for k, v in disagreement.items():
        print(f"  {k}: {v}")

    # ── Save ──────────────────────────────────────────────────
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Save raw predictions for error analysis
    np.savez(
        os.path.join(output_dir, "predictions.npz"),
        rob_probs=rob_test_probs, deb_probs=deb_test_probs,
        ensemble_preds=ensemble_preds, labels=y_test,
    )
    print(f"\nAll results saved to {output_dir}/")


if __name__ == "__main__":
    main()
