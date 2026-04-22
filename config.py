"""
Centralized Configuration — CENG 454 Group 2
Parameter-Efficient Fine-Tuning Ensembles for Sentiment Analysis

This file defines all hyperparameters and settings.
The notebook embeds a copy of these values, but this file
serves as the single source of truth for the submission package.
"""

# ── Reproducibility ──────────────────────────────────────────
SEED = 42

# ── Models ───────────────────────────────────────────────────
ROBERTA_MODEL = "roberta-base"
DEBERTA_MODEL = "microsoft/deberta-v3-base"

# ── Tokenization ────────────────────────────────────────────
MAX_LENGTH = 256          # Max token length per input

# ── LoRA ─────────────────────────────────────────────────────
LORA_R = 8                # Low-rank dimension
LORA_ALPHA = 16           # Scaling factor (alpha / r = 2)
LORA_DROPOUT = 0.1        # Dropout in LoRA layers

# Target attention modules per model
ROBERTA_TARGET_MODULES = ["query", "value"]
DEBERTA_TARGET_MODULES = ["query_proj", "value_proj"]

# ── Training ─────────────────────────────────────────────────
EPOCHS = 3
BATCH_SIZE = 16
EVAL_BATCH_SIZE = 32
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
FP16 = True               # Mixed precision (requires GPU)
LOGGING_STEPS = 100

# ── Ensemble ─────────────────────────────────────────────────
ENSEMBLE_VAL_SIZE = 5000   # Held-out split for meta-learner training

# ── Sentiment140 ─────────────────────────────────────────────
SENT140_SUBSAMPLE = 50000  # Subsample size (25k train + 25k test)

# ── Paths ────────────────────────────────────────────────────
OUTPUT_DIR = "./results"
CHECKPOINT_DIR = "./checkpoints"
