# CENG 454 — Parameter-Efficient Fine-Tuning Ensembles for Sentiment Analysis

**Group 2** | Artificial Intelligence and Data Science | Spring 2026

## Team
| Name | Student No |
|---|---|
| Mustafa Onur Bayram | 210401018 |
| Batuhan Türkaslan | 210401052 |
| Batuhan Bilecen | 210401002 |
| Efekan Kandak | 210401029 |

## Project Summary
We fine-tune two transformer models (RoBERTa, DeBERTa) using LoRA (Low-Rank Adaptation), then combine them via a stacking ensemble for binary sentiment classification. We compare this approach against a fully fine-tuned baseline to demonstrate that parameter-efficient ensembles achieve competitive or superior accuracy at a fraction of the computational cost.

## Datasets
- **IMDB Reviews** (primary): 50k movie reviews, binary sentiment. Loaded automatically via HuggingFace `datasets`.
- **Sentiment140** (secondary): 1.6M tweets, subsampled to 50k. Loaded automatically via HuggingFace `datasets`.

*Note: Datasets are downloaded dynamically during script execution. No manual data downloading is required.*

## Project Structure
```
AI-Project/
├── README.md                      # This file
├── requirements.txt               # Python dependencies
├── .gitignore                     # Git ignore rules
├── config.py                      # Centralized hyperparameters
├── src/                           # Source code
│   ├── __init__.py
│   ├── data_utils.py              # Data loading, splitting, tokenization
│   ├── train_baseline.py          # Full fine-tune RoBERTa
│   ├── train_lora.py              # LoRA fine-tune (RoBERTa or DeBERTa)
│   ├── train_ensemble.py          # Stacking ensemble
│   └── evaluate.py                # Tables, plots, error analysis
├── report/
│   └── report_outline.md          # Report structure and writing guide
├── results/                       # (Ignored) Saved metrics, plots, tables
└── checkpoints/                   # (Ignored) Saved model adapters
```

> [!IMPORTANT]
> The `results/` and `checkpoints/` directories are intentionally ignored by Git to prevent uploading large model weights and local experiment data. These directories will be created automatically when you run the scripts.

## Installation
1. Clone the repository:
```bash
git clone https://github.com/MustafaOnurBayram/AI-Project.git
cd AI-Project
```
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## How to Run (Local or Colab)
All scripts should be executed from the `src/` directory. If using Google Colab, make sure to set your runtime to GPU (T4).

```bash
# Navigate to source directory
cd src/

# 1. Train Baseline (~30-45 min on T4)
python train_baseline.py

# 2. Train LoRA RoBERTa on IMDB (~15-20 min)
python train_lora.py --model roberta --dataset imdb

# 3. Train LoRA DeBERTa on IMDB (~20-25 min)
python train_lora.py --model deberta --dataset imdb

# 4. Train Stacking Ensemble on IMDB (~5 min)
python train_ensemble.py --dataset imdb

# 5. Evaluate and Generate Plots (~2 min)
python evaluate.py
```

**(Optional) Run Sentiment140 Extended Pipeline:**
```bash
python train_lora.py --model roberta --dataset sentiment140
python train_lora.py --model deberta --dataset sentiment140
python train_ensemble.py --dataset sentiment140
python evaluate.py
```

## Submission Details
- **Deadline**: 22 May 2026, 23:59
- **Format**: `CENG454_Group_2.zip`
- **Platform**: UBYS

## AI Usage Disclosure
AI tools (GitHub Copilot, ChatGPT, etc.) were used for code scaffolding and debugging assistance. All experimental results are genuine and reproducible. All code was reviewed, understood, and modified by team members.
