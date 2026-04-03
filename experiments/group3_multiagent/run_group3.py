"""
Group 3 Runner: Multi-agent web verification experiments.

Usage:
    python -m experiments.group3_multiagent.run_group3 [--quick] [--num-samples 100]
"""

import os
import sys
import argparse
import re

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from experiments.config import ExperimentConfig, MultiAgentConfig
from experiments.metrics import compute_metrics, ResultLogger, plot_comparison_table
from experiments.group3_multiagent.orchestrator import VerificationPipeline


def load_test_data(cfg: ExperimentConfig, num_samples: int = 100):
    """Load a subset of test data for multi-agent evaluation."""
    base = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    csv_path = os.path.join(base, cfg.data.csv_path)
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["text", "label"])
    df["clean_text"] = df["text"].apply(
        lambda t: re.sub(r"\s+", " ", str(t)).strip() if isinstance(t, str) else ""
    )
    df = df[df["clean_text"].str.len() > 50].reset_index(drop=True)

    texts = df["clean_text"].values
    labels = df["label"].values.astype(int)

    _, X_test, _, y_test = train_test_split(
        texts, labels, test_size=cfg.data.test_size,
        random_state=cfg.data.random_state, stratify=labels,
    )

    # Sample subset for efficiency
    n = min(num_samples, len(X_test))
    indices = np.random.RandomState(42).choice(len(X_test), n, replace=False)
    return X_test[indices], y_test[indices]


def load_val_data(cfg: ExperimentConfig, num_samples: int = 50):
    """Load a validation split (different seed) for threshold / weight tuning."""
    base = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    csv_path = os.path.join(base, cfg.data.csv_path)
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["text", "label"])
    df["clean_text"] = df["text"].apply(
        lambda t: re.sub(r"\s+", " ", str(t)).strip() if isinstance(t, str) else ""
    )
    df = df[df["clean_text"].str.len() > 50].reset_index(drop=True)

    texts = df["clean_text"].values
    labels = df["label"].values.astype(int)

    _, X_test, _, y_test = train_test_split(
        texts, labels, test_size=cfg.data.test_size,
        random_state=cfg.data.random_state, stratify=labels,
    )

    n = min(num_samples, len(X_test))
    indices = np.random.RandomState(99).choice(len(X_test), n, replace=False)
    return X_test[indices], y_test[indices]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--judge-mode", choices=["rule", "llm"], default="rule")
    parser.add_argument("--tune", action="store_true",
                        help="Run grid search on val set to find best threshold/weights before eval")
    parser.add_argument("--tune-samples", type=int, default=40,
                        help="Number of val samples for grid search")
    args = parser.parse_args()

    print("=" * 60)
    print("  Group 3: Multi-Agent Web Verification (FactAgent)")
    print("=" * 60)

    output_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "outputs", "group3_multiagent",
    )
    os.makedirs(output_dir, exist_ok=True)
    logger = ResultLogger(os.path.join(output_dir, "results.csv"))

    num_samples = 20 if args.quick else args.num_samples

    cfg = ExperimentConfig(
        name=f"G3_factagent_{args.judge_mode}",
        group="group3_multiagent",
        multiagent=MultiAgentConfig(judge_mode=args.judge_mode),
    )

    print("  Building verification pipeline...")
    pipeline = VerificationPipeline.from_config(cfg.multiagent)

    # ---- Optional: grid-search threshold & weights on validation set ----
    if args.tune:
        from experiments.group3_multiagent.agents import JudgeAgent as JA
        print("\n  === Grid Search: finding best threshold & weights ===")
        val_texts, val_labels = load_val_data(cfg, num_samples=args.tune_samples)
        best_t, best_w, _ = JA.grid_search(
            list(val_texts), val_labels, pipeline,
        )
        pipeline.judge.fake_threshold = best_t
        pipeline.judge.tool_weights = best_w
        print(f"  Applied: threshold={best_t}, weights={best_w}\n")

    # ---- Main evaluation on test set ----
    print("  Loading test data...")
    texts, labels = load_test_data(cfg, num_samples=num_samples)
    print(f"  Evaluating {len(texts)} samples")

    predictions = []
    confidences = []

    for i, (text, label) in enumerate(zip(texts, labels)):
        result = pipeline.verify(text)
        predictions.append(result.prediction)
        confidences.append(result.confidence)
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  [{i+1}/{len(texts)}] pred={result.prediction} "
                  f"true={label} conf={result.confidence:.2f}")

    predictions = np.array(predictions)
    eval_result = compute_metrics(
        labels, predictions, np.array(confidences),
        experiment_name=cfg.name, group=cfg.group,
    )

    print(f"\n  {eval_result.summary()}")
    logger.log(eval_result)

    plot_comparison_table(
        [eval_result],
        save_path=os.path.join(output_dir, "group3_comparison.png"),
        title="Group 3: FactAgent Multi-Tool Verification",
    )

    print(f"\n  Group 3 results saved to {output_dir}")


if __name__ == "__main__":
    main()
