"""
Data loading, splitting, and tokenization utilities.
CENG 454 — Group 2
"""
import os
import random
import numpy as np
import torch
from datasets import load_dataset, Dataset, DatasetDict
from transformers import AutoTokenizer

# ── Project root (one level up from src/) ────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
CHECKPOINTS_DIR = os.path.join(PROJECT_ROOT, "checkpoints")


# ── Reproducibility ──────────────────────────────────────────
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        gpu_mem = torch.cuda.get_device_properties(0).total_mem / 1e9
        print(f"VRAM: {gpu_mem:.1f} GB")
    return device


# ── IMDB ─────────────────────────────────────────────────────
def load_imdb(ensemble_val_size=5000, seed=42):
    """
    Load IMDB and split training set into:
      - train_base (20k) for model training
      - val_ensemble (5k) for meta-learner training
      - test (25k) for final evaluation
    """
    ds = load_dataset("imdb")
    print(f"IMDB loaded: {len(ds['train'])} train, {len(ds['test'])} test")

    # Split train into base-train + ensemble-val
    split = ds["train"].train_test_split(
        test_size=ensemble_val_size, seed=seed, stratify_by_column="label"
    )
    return DatasetDict({
        "train": split["train"],          # 20,000
        "val_ensemble": split["test"],    # 5,000
        "test": ds["test"],               # 25,000
    })


# ── Sentiment140 ─────────────────────────────────────────────
def load_sentiment140(subsample=50000, seed=42):
    """
    Load Sentiment140, map labels (0→0, 4→1), subsample, and split 50/50.
    """
    ds = load_dataset("sentiment140", split="train")
    print(f"Sentiment140 loaded: {len(ds)} samples")

    # The dataset columns are: 'text', 'sentiment', 'date', 'user', 'query'
    # Map labels: 0=negative stays 0, 4=positive becomes 1
    ds = ds.map(lambda x: {"label": 1 if x["sentiment"] == 4 else 0})
    ds = ds.remove_columns([c for c in ds.column_names if c not in ("text", "label")])

    # Stratified subsample
    ds = ds.shuffle(seed=seed).select(range(min(subsample, len(ds))))
    split = ds.train_test_split(test_size=0.5, seed=seed, stratify_by_column="label")

    # Further split train for ensemble val (10%)
    train_split = split["train"].train_test_split(
        test_size=0.1, seed=seed, stratify_by_column="label"
    )
    return DatasetDict({
        "train": train_split["train"],
        "val_ensemble": train_split["test"],
        "test": split["test"],
    })


# ── Tokenization ─────────────────────────────────────────────
def tokenize_dataset(dataset_dict, model_name, max_length=256):
    """
    Tokenize all splits with the tokenizer for `model_name`.
    Accepts either a DatasetDict or a plain dict of Datasets.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tok_fn(batch):
        return tokenizer(
            batch["text"], padding="max_length",
            truncation=True, max_length=max_length
        )

    tokenized = {}
    items = dataset_dict.items()
    for split_name, split_ds in items:
        tokenized[split_name] = split_ds.map(tok_fn, batched=True, batch_size=1000)
        tokenized[split_name].set_format(
            "torch", columns=["input_ids", "attention_mask", "label"]
        )
    return DatasetDict(tokenized), tokenizer


if __name__ == "__main__":
    set_seed(42)
    print(f"Project root: {PROJECT_ROOT}")
    imdb = load_imdb()
    for k, v in imdb.items():
        print(f"  {k}: {len(v)} samples")
