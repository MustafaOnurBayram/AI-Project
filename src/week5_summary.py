"""
Final Summary: Aggregates all results, creates comparison tables,
generates plots and exports everything to results/ folder.
CENG 454 — Group 2

Usage:
    python week5_summary.py
    
Run this AFTER all other weeks have completed successfully.
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def load_json(filename):
    """Load a JSON file from results/ directory."""
    path = os.path.join(RESULTS_DIR, filename)
    if not os.path.exists(path):
        print(f"  WARNING: {filename} not found — skipping")
        return None
    with open(path) as f:
        return json.load(f)


def main():
    print("=" * 60)
    print("  FINAL PROJECT SUMMARY — CENG 454 Group 2")
    print("=" * 60)

    # ─── Load all results ─────────────────────────────────────
    w1 = load_json("week1_baseline.json")
    w2_rob = load_json("week2_roberta_lora.json")
    w2_deb = load_json("week2_deberta_lora.json")
    w3_ens = load_json("week3_ensemble_2_3.json")
    w3_full = load_json("week3_ensemble_full.json")
    w4 = load_json("week4_cross_domain.json")

    # ─── In-Domain Comparison Table ───────────────────────────
    in_domain_models = []
    for label, data in [
        ("1. SBERT + LogReg", w1),
        ("2. RoBERTa + LoRA", w2_rob),
        ("3. DeBERTa + LoRA", w2_deb),
        ("4. Stacked (2+3)", w3_ens),
        ("5. Full Stack (1+2+3)", w3_full),
    ]:
        if data:
            in_domain_models.append({
                "Model": label,
                "Accuracy": data.get("accuracy", "N/A"),
                "F1": data.get("f1", "N/A"),
                "Precision": data.get("precision", "N/A"),
                "Recall": data.get("recall", "N/A"),
            })

    if in_domain_models:
        print("\n" + "=" * 70)
        print("  TABLE 1: IN-DOMAIN PERFORMANCE (ErfanMoosaviMonazzah Dataset)")
        print("=" * 70)
        header = f"{'Model':<28} | {'Accuracy':>9} | {'F1':>9} | {'Precision':>10} | {'Recall':>7}"
        print(header)
        print("-" * len(header))
        for m in in_domain_models:
            print(f"{m['Model']:<28} | {m['Accuracy']:>8}% | {m['F1']:>8}% | {m['Precision']:>9}% | {m['Recall']:>6}%")
        print("=" * 70)

    # ─── Parameter Efficiency Table ───────────────────────────
    param_models = []
    for label, data in [
        ("1. SBERT + LogReg", w1),
        ("2. RoBERTa + LoRA", w2_rob),
        ("3. DeBERTa + LoRA", w2_deb),
    ]:
        if data and "trainable_params" in data:
            param_models.append({
                "Model": label,
                "Trainable Params": f"{data['trainable_params']:,}" if data['trainable_params'] else "0 (pretrained)",
                "Training Time": f"{data.get('training_time_seconds', 0) / 60:.1f} min",
                "Accuracy": data.get("accuracy", "N/A"),
            })

    if param_models:
        print("\n" + "=" * 70)
        print("  TABLE 2: PARAMETER EFFICIENCY COMPARISON")
        print("=" * 70)
        header = f"{'Model':<28} | {'Trainable Params':>18} | {'Train Time':>12} | {'Accuracy':>9}"
        print(header)
        print("-" * len(header))
        for m in param_models:
            print(f"{m['Model']:<28} | {m['Trainable Params']:>18} | {m['Training Time']:>12} | {m['Accuracy']:>8}%")
        print("=" * 70)

    # ─── Cross-Domain Results ─────────────────────────────────
    if w4:
        print("\n" + "=" * 70)
        print(f"  TABLE 3: CROSS-DOMAIN PERFORMANCE ({w4['dataset']}, n={w4['num_samples']})")
        print("=" * 70)
        header = f"{'Model':<28} | {'Cross-Domain Acc':>17} | {'Cross-Domain F1':>16}"
        print(header)
        print("-" * len(header))
        for m in w4["models"]:
            print(f"{m['name']:<28} | {m['accuracy']:>16}% | {m['f1']:>15}%")
        print("=" * 70)

    # ─── Generalization Gap Table ─────────────────────────────
    if w4 and in_domain_models:
        print("\n" + "=" * 70)
        print("  TABLE 4: GENERALIZATION GAP (In-Domain vs Cross-Domain)")
        print("=" * 70)
        header = f"{'Model':<28} | {'In-Domain':>10} | {'Cross-Dom':>10} | {'Gap':>7}"
        print(header)
        print("-" * len(header))
        for i, cd_model in enumerate(w4["models"]):
            if i < len(in_domain_models):
                in_acc = in_domain_models[i]["Accuracy"]
                cd_acc = cd_model["accuracy"]
                if isinstance(in_acc, (int, float)) and isinstance(cd_acc, (int, float)):
                    gap = round(cd_acc - in_acc, 2)
                    print(f"{cd_model['name']:<28} | {in_acc:>9}% | {cd_acc:>9}% | {gap:>+6}%")
        print("=" * 70)

    # ─── Meta-Learner Weights ─────────────────────────────────
    if w3_full and "meta_weights" in w3_full:
        print("\n" + "=" * 50)
        print("  META-LEARNER WEIGHTS (Full Stack)")
        print("=" * 50)
        for k, v in w3_full["meta_weights"].items():
            bar = "█" * int(abs(v) * 10)
            print(f"  {k:<10}: {v:>8.4f}  {bar}")
        print("=" * 50)

    # ─── Generate Plots ───────────────────────────────────────
    if in_domain_models:
        # Plot 1: In-Domain Accuracy Comparison Bar Chart
        names = [m["Model"] for m in in_domain_models]
        accs = [m["Accuracy"] for m in in_domain_models]

        fig, ax = plt.subplots(figsize=(10, 5))
        colors = sns.color_palette("viridis", len(names))
        bars = ax.bar(names, accs, color=colors)
        ax.set_ylabel("Accuracy (%)")
        ax.set_title("In-Domain Accuracy Comparison")
        ax.set_ylim(min(accs) - 5, max(accs) + 3)
        for bar, val in zip(bars, accs):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    f"{val}%", ha="center", va="bottom", fontsize=9, fontweight="bold")
        plt.xticks(rotation=20, ha="right")
        plt.tight_layout()
        path = os.path.join(RESULTS_DIR, "plot_accuracy_comparison.png")
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"\n✓ Plot saved: {path}")

        # Plot 2: F1 Score Comparison
        f1s = [m["F1"] for m in in_domain_models]
        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.bar(names, f1s, color=sns.color_palette("magma", len(names)))
        ax.set_ylabel("F1 Score (%)")
        ax.set_title("In-Domain F1 Score Comparison")
        ax.set_ylim(min(f1s) - 5, max(f1s) + 3)
        for bar, val in zip(bars, f1s):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    f"{val}%", ha="center", va="bottom", fontsize=9, fontweight="bold")
        plt.xticks(rotation=20, ha="right")
        plt.tight_layout()
        path = os.path.join(RESULTS_DIR, "plot_f1_comparison.png")
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"✓ Plot saved: {path}")

    # Plot 3: Cross-Domain vs In-Domain (Grouped Bar)
    if w4 and in_domain_models:
        cd_names = [m["name"] for m in w4["models"]]
        cd_accs = [m["accuracy"] for m in w4["models"]]
        id_accs = [in_domain_models[i]["Accuracy"] for i in range(min(len(cd_names), len(in_domain_models)))]

        x = np.arange(len(cd_names))
        width = 0.35

        fig, ax = plt.subplots(figsize=(11, 5))
        bars1 = ax.bar(x - width/2, id_accs, width, label="In-Domain", color="#4C72B0")
        bars2 = ax.bar(x + width/2, cd_accs, width, label="Cross-Domain", color="#DD8452")
        ax.set_ylabel("Accuracy (%)")
        ax.set_title("Generalization: In-Domain vs Cross-Domain Accuracy")
        ax.set_xticks(x)
        ax.set_xticklabels(cd_names, rotation=20, ha="right")
        ax.legend()
        ax.set_ylim(0, max(max(id_accs), max(cd_accs)) + 8)
        for bar, val in zip(bars1, id_accs):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f"{val}%", ha="center", fontsize=8)
        for bar, val in zip(bars2, cd_accs):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f"{val}%", ha="center", fontsize=8)
        plt.tight_layout()
        path = os.path.join(RESULTS_DIR, "plot_generalization_gap.png")
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"✓ Plot saved: {path}")

    # Plot 4: Parameter Efficiency Scatter
    if w2_rob and w2_deb and w1:
        models_pe = [
            ("SBERT+LogReg", 0, w1.get("accuracy", 0)),
            ("RoBERTa+LoRA", w2_rob.get("trainable_params", 0), w2_rob.get("accuracy", 0)),
            ("DeBERTa+LoRA", w2_deb.get("trainable_params", 0), w2_deb.get("accuracy", 0)),
        ]
        fig, ax = plt.subplots(figsize=(8, 5))
        for name, params, acc in models_pe:
            ax.scatter(params / 1e6, acc, s=150, zorder=3)
            ax.annotate(name, (params / 1e6, acc), textcoords="offset points",
                        xytext=(8, 5), fontsize=10)
        ax.set_xlabel("Trainable Parameters (Millions)")
        ax.set_ylabel("Accuracy (%)")
        ax.set_title("Parameter Efficiency: Accuracy vs Trainable Params")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        path = os.path.join(RESULTS_DIR, "plot_param_efficiency.png")
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"✓ Plot saved: {path}")

    # ─── Save combined summary JSON ───────────────────────────
    summary = {
        "in_domain": in_domain_models,
        "cross_domain": w4 if w4 else {},
    }
    summary_path = os.path.join(RESULTS_DIR, "final_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n✓ Combined summary saved to {summary_path}")

    # ─── Final ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  ALL RESULTS SAVED TO:", RESULTS_DIR)
    print("=" * 60)
    print("\nFiles generated:")
    for f in sorted(os.listdir(RESULTS_DIR)):
        size = os.path.getsize(os.path.join(RESULTS_DIR, f))
        print(f"  📄 {f} ({size:,} bytes)")
    print("\n✅ Done! You can download the results/ folder to keep as backup.")


if __name__ == "__main__":
    main()
