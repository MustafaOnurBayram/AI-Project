"""
Evaluation, visualization, and error analysis.
CENG 454 — Group 2

Usage:
    python evaluate.py
"""
import json
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend — works in scripts and Colab terminal
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay
)
from data_utils import set_seed, load_imdb, RESULTS_DIR

SEED = 42
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")


def load_all_metrics():
    """Load metrics.json from each experiment directory."""
    experiments = {}
    for d in os.listdir(RESULTS_DIR):
        metrics_path = os.path.join(RESULTS_DIR, d, "metrics.json")
        if os.path.isfile(metrics_path):
            with open(metrics_path) as f:
                experiments[d] = json.load(f)
    return experiments


def build_comparison_table(experiments):
    """Build a DataFrame comparing all experiments."""
    rows = []
    for name, m in experiments.items():
        rows.append({
            "Experiment": name,
            "Accuracy": m.get("eval_accuracy", m.get("accuracy", None)),
            "F1": m.get("eval_f1", m.get("f1", None)),
            "Precision": m.get("eval_precision", m.get("precision", None)),
            "Recall": m.get("eval_recall", m.get("recall", None)),
            "Train Time (s)": m.get("training_time_seconds", None),
            "Trainable Params": m.get("trainable_params", None),
        })
    df = pd.DataFrame(rows)
    return df


def plot_accuracy_comparison(df):
    """Bar chart comparing accuracy across models."""
    os.makedirs(PLOTS_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(df["Experiment"], df["Accuracy"],
                  color=sns.color_palette("viridis", len(df)))
    ax.set_ylabel("Accuracy")
    ax.set_title("Model Accuracy Comparison")
    ax.set_ylim(0.8, 1.0)
    plt.xticks(rotation=30, ha="right")
    for bar, val in zip(bars, df["Accuracy"]):
        if val is not None:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                    f"{val:.4f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "accuracy_comparison.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def plot_training_time(df):
    """Bar chart comparing training times."""
    os.makedirs(PLOTS_DIR, exist_ok=True)
    df_time = df.dropna(subset=["Train Time (s)"])
    if df_time.empty:
        print("No training time data to plot.")
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(df_time["Experiment"], df_time["Train Time (s)"] / 60,
                  color=sns.color_palette("magma", len(df_time)))
    ax.set_ylabel("Training Time (minutes)")
    ax.set_title("Training Time Comparison")
    plt.xticks(rotation=30, ha="right")
    for bar, val in zip(bars, df_time["Train Time (s)"] / 60):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                f"{val:.1f}m", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "training_time.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def plot_param_efficiency(df):
    """Scatter: trainable params vs accuracy."""
    os.makedirs(PLOTS_DIR, exist_ok=True)
    df_p = df.dropna(subset=["Trainable Params", "Accuracy"])
    if df_p.empty:
        print("No param/accuracy data to plot.")
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(df_p["Trainable Params"] / 1e6, df_p["Accuracy"], s=120, zorder=3)
    for _, row in df_p.iterrows():
        ax.annotate(row["Experiment"],
                     (row["Trainable Params"]/1e6, row["Accuracy"]),
                     textcoords="offset points", xytext=(5, 5), fontsize=8)
    ax.set_xlabel("Trainable Parameters (Millions)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Parameter Efficiency: Accuracy vs. Trainable Params")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "param_efficiency.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def error_analysis(dataset_tag="imdb"):
    """Analyze misclassified examples from the ensemble."""
    pred_path = os.path.join(RESULTS_DIR, f"ensemble_{dataset_tag}", "predictions.npz")
    if not os.path.exists(pred_path):
        print(f"No predictions found at {pred_path}. Run ensemble first.")
        return

    data = np.load(pred_path)
    preds = data["ensemble_preds"]
    labels = data["labels"]

    # Load text for error examples
    if dataset_tag == "imdb":
        ds = load_imdb(seed=SEED)
        texts = ds["test"]["text"]
    else:
        print("Sentiment140 error analysis: load dataset manually.")
        return

    # Find misclassified
    wrong_idx = np.where(preds != labels)[0]
    print(f"\nTotal misclassified: {len(wrong_idx)} / {len(labels)}")
    print(f"Error rate: {len(wrong_idx)/len(labels)*100:.2f}%\n")

    # Show first 10 errors
    print("=== Sample Misclassifications ===")
    for i, idx in enumerate(wrong_idx[:10]):
        print(f"\n--- Error {i+1} ---")
        print(f"  True: {'positive' if labels[idx]==1 else 'negative'}")
        print(f"  Pred: {'positive' if preds[idx]==1 else 'negative'}")
        text_preview = texts[idx][:200].replace('\n', ' ')
        print(f"  Text: {text_preview}...")

    # Confusion matrix
    os.makedirs(PLOTS_DIR, exist_ok=True)
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(cm, display_labels=["Negative", "Positive"]).plot(ax=ax)
    ax.set_title(f"Ensemble Confusion Matrix ({dataset_tag.upper()})")
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, f"confusion_{dataset_tag}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")

    # Classification report
    report = classification_report(labels, preds,
                                    target_names=["Negative", "Positive"])
    print(f"\n=== Classification Report ===\n{report}")
    report_path = os.path.join(
        RESULTS_DIR, f"ensemble_{dataset_tag}", "classification_report.txt"
    )
    with open(report_path, "w") as f:
        f.write(report)


def main():
    set_seed(SEED)
    print("=" * 60)
    print("  EVALUATION & ANALYSIS")
    print("=" * 60)

    # Load all experiment metrics
    experiments = load_all_metrics()
    if not experiments:
        print("No results found in", RESULTS_DIR)
        print("Run training scripts first.")
        return

    df = build_comparison_table(experiments)
    print("\n=== Results Comparison Table ===")
    print(df.to_string(index=False))

    # Save table
    os.makedirs(PLOTS_DIR, exist_ok=True)
    csv_path = os.path.join(RESULTS_DIR, "comparison_table.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")

    # Plots
    plot_accuracy_comparison(df)
    plot_training_time(df)
    plot_param_efficiency(df)

    # Error analysis
    error_analysis("imdb")

    print("\n✓ All evaluation artifacts saved to", RESULTS_DIR)


if __name__ == "__main__":
    main()
