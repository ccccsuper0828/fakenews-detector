"""
Unified evaluation dashboard for all experiment groups.

Collects results from all groups, generates comparison tables,
charts, and a comprehensive analysis report.

Usage:
    python -m experiments.evaluate_all [--output-dir outputs/evaluation]
"""

import os
import sys
import csv
import json
import argparse
from typing import List, Dict

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

matplotlib.rcParams["font.family"] = ["Arial", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from experiments.metrics import EvalResult


def load_results_from_csv(csv_path: str) -> List[Dict]:
    """Load experiment results from a CSV log file."""
    results = []
    if not os.path.exists(csv_path):
        return results
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key in ["accuracy", "precision", "recall", "f1", "auc_roc", "inference_time_ms"]:
                if key in row:
                    row[key] = float(row[key]) if row[key] else 0.0
            if "param_count" in row:
                row["param_count"] = int(row["param_count"]) if row["param_count"] else 0
            results.append(row)
    return results


def collect_all_results(base_output_dir: str) -> List[Dict]:
    """Collect results from all experiment group CSV logs."""
    all_results = []
    group_dirs = [
        "group1_rnn", "group2_llm", "group3_multiagent", "group4_kg_rl",
    ]

    for group_dir in group_dirs:
        csv_path = os.path.join(base_output_dir, group_dir, "results.csv")
        results = load_results_from_csv(csv_path)
        all_results.extend(results)
        if results:
            print(f"  Loaded {len(results)} results from {group_dir}")

    return all_results


def plot_comprehensive_comparison(results: List[Dict], save_dir: str):
    """Generate comprehensive comparison charts."""
    if not results:
        print("  No results to plot")
        return

    names = [r["experiment_name"] for r in results]
    groups = [r.get("group", "unknown") for r in results]
    accs = [r["accuracy"] for r in results]
    f1s = [r["f1"] for r in results]
    aucs = [r["auc_roc"] for r in results]
    params = [r["param_count"] for r in results]

    # 1. Bar chart: F1 comparison across all experiments
    fig, ax = plt.subplots(figsize=(max(12, len(names) * 0.8), 7))
    colors = []
    group_colors = {
        "group1_rnn": "#4c72b0",
        "group2_llm": "#55a868",
        "group3_multiagent": "#c44e52",
        "group4_kg_rl": "#8172b2",
    }
    for g in groups:
        colors.append(group_colors.get(g, "#666666"))

    bars = ax.bar(range(len(names)), f1s, color=colors, alpha=0.85)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=60, ha="right", fontsize=8)
    ax.set_ylabel("F1 Score")
    ax.set_title("F1 Score Comparison Across All Experiments", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)

    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, label=g.replace("_", " ").title())
                       for g, c in group_colors.items()
                       if g in groups]
    ax.legend(handles=legend_elements, loc="lower right")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "all_f1_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # 2. Grouped bar: Acc, F1, AUC side by side
    fig, ax = plt.subplots(figsize=(max(14, len(names) * 1.2), 7))
    x = np.arange(len(names))
    width = 0.25
    ax.bar(x - width, accs, width, label="Accuracy", color="#4c72b0", alpha=0.85)
    ax.bar(x, f1s, width, label="F1", color="#55a868", alpha=0.85)
    ax.bar(x + width, aucs, width, label="AUC-ROC", color="#c44e52", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=60, ha="right", fontsize=8)
    ax.set_ylabel("Score")
    ax.set_title("Multi-Metric Comparison", fontsize=14, fontweight="bold")
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "all_metrics_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # 3. Scatter: F1 vs Parameter count (efficiency)
    fig, ax = plt.subplots(figsize=(10, 7))
    for i, (name, f1, pc, g) in enumerate(zip(names, f1s, params, groups)):
        color = group_colors.get(g, "#666666")
        ax.scatter(pc, f1, c=color, s=100, alpha=0.8, zorder=5)
        ax.annotate(name, (pc, f1), fontsize=7, ha="left", va="bottom",
                    xytext=(5, 5), textcoords="offset points")

    ax.set_xlabel("Trainable Parameters")
    ax.set_ylabel("F1 Score")
    ax.set_title("Model Efficiency: F1 vs Parameters", fontsize=14, fontweight="bold")
    ax.grid(alpha=0.3)
    if max(params) > 0:
        ax.set_xscale("log")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "efficiency_scatter.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # 4. Group-level summary (best per group)
    group_best = {}
    for r in results:
        g = r.get("group", "unknown")
        if g not in group_best or r["f1"] > group_best[g]["f1"]:
            group_best[g] = r

    if group_best:
        fig, ax = plt.subplots(figsize=(10, 6))
        group_names = list(group_best.keys())
        best_names = [group_best[g]["experiment_name"] for g in group_names]
        best_f1 = [group_best[g]["f1"] for g in group_names]
        best_acc = [group_best[g]["accuracy"] for g in group_names]

        x = np.arange(len(group_names))
        ax.bar(x - 0.15, best_acc, 0.3, label="Accuracy", color="#4c72b0")
        ax.bar(x + 0.15, best_f1, 0.3, label="F1", color="#55a868")

        display_names = [f"{gn}\n({bn})" for gn, bn in zip(group_names, best_names)]
        ax.set_xticks(x)
        ax.set_xticklabels(display_names, fontsize=9)
        ax.set_ylabel("Score")
        ax.set_title("Best Model Per Group", fontsize=14, fontweight="bold")
        ax.legend()
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "group_best_comparison.png"), dpi=150, bbox_inches="tight")
        plt.close()

    print(f"  Charts saved to {save_dir}")


def generate_report(results: List[Dict], save_dir: str):
    """Generate a text-based analysis report."""
    report_lines = [
        "=" * 60,
        "  FAKE NEWS DETECTION - EXPERIMENT RESULTS REPORT",
        "=" * 60,
        "",
    ]

    if not results:
        report_lines.append("  No results available yet. Run experiments first.")
    else:
        # Summary table
        report_lines.append(f"  Total experiments: {len(results)}")
        report_lines.append("")
        report_lines.append(
            f"  {'Experiment':<35s} {'Group':<18s} {'Acc':>7s} {'F1':>7s} "
            f"{'AUC':>7s} {'Params':>12s}"
        )
        report_lines.append("  " + "-" * 90)

        for r in sorted(results, key=lambda x: x["f1"], reverse=True):
            report_lines.append(
                f"  {r['experiment_name']:<35s} {r.get('group',''):<18s} "
                f"{r['accuracy']:>7.4f} {r['f1']:>7.4f} "
                f"{r['auc_roc']:>7.4f} {r['param_count']:>12,d}"
            )

        # Best per group
        report_lines.append("")
        report_lines.append("  Best model per group:")
        report_lines.append("  " + "-" * 60)
        group_best = {}
        for r in results:
            g = r.get("group", "unknown")
            if g not in group_best or r["f1"] > group_best[g]["f1"]:
                group_best[g] = r
        for g, r in group_best.items():
            report_lines.append(
                f"  {g:<20s}: {r['experiment_name']} "
                f"(F1={r['f1']:.4f}, Acc={r['accuracy']:.4f})"
            )

        # Overall best
        best = max(results, key=lambda x: x["f1"])
        report_lines.append("")
        report_lines.append(f"  Overall best: {best['experiment_name']} (F1={best['f1']:.4f})")

    report_lines.append("")
    report_lines.append("=" * 60)

    report_text = "\n".join(report_lines)
    print(report_text)

    report_path = os.path.join(save_dir, "experiment_report.txt")
    with open(report_path, "w") as f:
        f.write(report_text)
    print(f"\n  Report saved to {report_path}")

    # Also save as JSON
    json_path = os.path.join(save_dir, "all_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  JSON results saved to {json_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    base = os.path.dirname(os.path.dirname(__file__))
    base_output = os.path.join(base, "outputs")
    eval_dir = args.output_dir or os.path.join(base_output, "evaluation")
    os.makedirs(eval_dir, exist_ok=True)

    print("=" * 60)
    print("  Collecting results from all experiment groups...")
    print("=" * 60)

    all_results = collect_all_results(base_output)

    print(f"\n  Total results collected: {len(all_results)}")

    generate_report(all_results, eval_dir)
    plot_comprehensive_comparison(all_results, eval_dir)

    print(f"\n  Evaluation complete. All outputs in {eval_dir}")


if __name__ == "__main__":
    main()
