"""
Group 1 Runner: RNN ablation experiments.

Usage:
    python -m experiments.group1_rnn.run_group1 [--quick]
"""

import os
import sys
import itertools
import argparse

import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from experiments.config import ExperimentConfig, RNNConfig, TrainConfig, DataConfig
from experiments.runner import run_training
from experiments.metrics import ResultLogger, plot_comparison_table
from experiments.group1_rnn.models import RNNClassifier
from src.data_utils import load_and_preprocess_multi_data, load_glove_embeddings


def load_data(cfg: ExperimentConfig):
    dc = cfg.data
    base = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    csv_path = os.path.join(base, dc.csv_path)
    extra_dir = os.path.join(base, dc.extra_dataset_dir) if dc.extra_dataset_dir else None
    glove_path = os.path.join(base, "glove.6B.100d.txt")

    train_ds, val_ds, test_ds, vocab = load_and_preprocess_multi_data(
        csv_path,
        extra_dataset_dir=extra_dir,
        max_vocab_size=dc.max_vocab_size,
        max_length=dc.max_length,
        augment=dc.augment,
        num_aug=dc.num_aug,
        random_state=dc.random_state,
    )

    glove_matrix = None
    if os.path.exists(glove_path):
        glove_matrix, coverage = load_glove_embeddings(glove_path, vocab, embed_dim=100)
        print(f"  GloVe loaded, coverage: {coverage:.2%}")

    return train_ds, val_ds, test_ds, vocab, glove_matrix


def build_model(rc: RNNConfig, vocab_size: int, glove_matrix=None):
    emb = glove_matrix if rc.embedding_type != "random" else None
    return RNNClassifier(
        vocab_size=vocab_size,
        embed_dim=rc.embed_dim,
        hidden_dim=rc.hidden_dim,
        num_layers=rc.num_layers,
        rnn_type=rc.rnn_type,
        attention_type=rc.attention_type,
        pooling=rc.pooling,
        num_attention_heads=rc.num_attention_heads,
        dropout=rc.dropout,
        pretrained_embeddings=emb,
        freeze_embeddings=rc.freeze_embeddings,
    )


def run_architecture_sweep(train_ds, val_ds, test_ds, vocab, glove_matrix, quick=False):
    """Sweep over RNN types and attention/pooling strategies."""
    rnn_types = ["lstm", "bilstm", "gru", "bigru"]
    attention_pooling = [
        ("additive", "attention"),
        ("dot", "attention"),
        ("multihead", "attention"),
        ("none", "mean"),
    ]

    if quick:
        rnn_types = ["bilstm", "gru"]
        attention_pooling = [("additive", "attention"), ("none", "mean")]

    results = []
    for rnn_type, (attn, pool) in itertools.product(rnn_types, attention_pooling):
        name = f"G1_{rnn_type}_{attn}_{pool}"
        print(f"\n{'='*60}\n  Experiment: {name}\n{'='*60}")

        cfg = ExperimentConfig(
            name=name, group="group1_rnn",
            train=TrainConfig(num_epochs=10 if quick else 20, batch_size=32, learning_rate=1e-3),
            rnn=RNNConfig(rnn_type=rnn_type, attention_type=attn, pooling=pool),
        )

        model = build_model(cfg.rnn, vocab.vocab_size, glove_matrix)
        history, model, result = run_training(
            model, train_ds, val_ds, test_ds, cfg
        )
        results.append(result)

    return results


def run_hyperparameter_sweep(train_ds, val_ds, test_ds, vocab, glove_matrix, quick=False):
    """Sweep LR, BS, hidden_dim, num_layers on best architecture (BiLSTM+additive)."""
    results = []

    lrs = [1e-2, 1e-3, 1e-4] if quick else [1e-2, 1e-3, 1e-4, 1e-5]
    for lr in lrs:
        name = f"G1_bilstm_lr{lr}"
        print(f"\n  Experiment: {name}")
        cfg = ExperimentConfig(
            name=name, group="group1_rnn",
            train=TrainConfig(num_epochs=10 if quick else 20, learning_rate=lr),
            rnn=RNNConfig(rnn_type="bilstm", attention_type="additive", pooling="attention"),
        )
        model = build_model(cfg.rnn, vocab.vocab_size, glove_matrix)
        _, _, result = run_training(model, train_ds, val_ds, test_ds, cfg)
        results.append(result)

    batch_sizes = [16, 64] if quick else [16, 32, 64, 128]
    for bs in batch_sizes:
        name = f"G1_bilstm_bs{bs}"
        print(f"\n  Experiment: {name}")
        cfg = ExperimentConfig(
            name=name, group="group1_rnn",
            train=TrainConfig(num_epochs=10 if quick else 20, batch_size=bs),
            rnn=RNNConfig(rnn_type="bilstm", attention_type="additive", pooling="attention"),
        )
        model = build_model(cfg.rnn, vocab.vocab_size, glove_matrix)
        _, _, result = run_training(model, train_ds, val_ds, test_ds, cfg)
        results.append(result)

    if not quick:
        for hd in [64, 128, 256]:
            name = f"G1_bilstm_hd{hd}"
            print(f"\n  Experiment: {name}")
            cfg = ExperimentConfig(
                name=name, group="group1_rnn",
                train=TrainConfig(num_epochs=20),
                rnn=RNNConfig(rnn_type="bilstm", hidden_dim=hd),
            )
            model = build_model(cfg.rnn, vocab.vocab_size, glove_matrix)
            _, _, result = run_training(model, train_ds, val_ds, test_ds, cfg)
            results.append(result)

        for nl in [1, 2, 3]:
            name = f"G1_bilstm_nl{nl}"
            print(f"\n  Experiment: {name}")
            cfg = ExperimentConfig(
                name=name, group="group1_rnn",
                train=TrainConfig(num_epochs=20),
                rnn=RNNConfig(rnn_type="bilstm", num_layers=nl),
            )
            model = build_model(cfg.rnn, vocab.vocab_size, glove_matrix)
            _, _, result = run_training(model, train_ds, val_ds, test_ds, cfg)
            results.append(result)

    return results


def run_embedding_sweep(train_ds, val_ds, test_ds, vocab, glove_matrix, quick=False):
    """Compare random vs GloVe embeddings, frozen vs fine-tuned."""
    results = []
    configs = [
        ("random_finetune", "random", False),
        ("glove100_frozen", "glove100", True),
        ("glove100_finetune", "glove100", False),
    ]

    for tag, emb_type, freeze in configs:
        name = f"G1_bilstm_{tag}"
        print(f"\n  Experiment: {name}")
        cfg = ExperimentConfig(
            name=name, group="group1_rnn",
            train=TrainConfig(num_epochs=10 if quick else 20),
            rnn=RNNConfig(
                rnn_type="bilstm", embedding_type=emb_type,
                freeze_embeddings=freeze,
            ),
        )
        model = build_model(cfg.rnn, vocab.vocab_size, glove_matrix)
        _, _, result = run_training(model, train_ds, val_ds, test_ds, cfg)
        results.append(result)

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Run reduced sweep for testing")
    args = parser.parse_args()

    print("=" * 60)
    print("  Group 1: RNN Ablation Experiments")
    print("=" * 60)

    cfg = ExperimentConfig()
    train_ds, val_ds, test_ds, vocab, glove_matrix = load_data(cfg)

    output_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "outputs", "group1_rnn"
    )
    os.makedirs(output_dir, exist_ok=True)
    logger = ResultLogger(os.path.join(output_dir, "results.csv"))

    all_results = []

    print("\n--- Architecture Sweep ---")
    arch_results = run_architecture_sweep(
        train_ds, val_ds, test_ds, vocab, glove_matrix, quick=args.quick
    )
    all_results.extend(arch_results)
    logger.log_many(arch_results)

    print("\n--- Hyperparameter Sweep ---")
    hp_results = run_hyperparameter_sweep(
        train_ds, val_ds, test_ds, vocab, glove_matrix, quick=args.quick
    )
    all_results.extend(hp_results)
    logger.log_many(hp_results)

    print("\n--- Embedding Sweep ---")
    emb_results = run_embedding_sweep(
        train_ds, val_ds, test_ds, vocab, glove_matrix, quick=args.quick
    )
    all_results.extend(emb_results)
    logger.log_many(emb_results)

    plot_comparison_table(
        all_results,
        save_path=os.path.join(output_dir, "group1_comparison.png"),
        title="Group 1: RNN Architecture Comparison",
    )

    print(f"\n  All Group 1 results saved to {output_dir}")


if __name__ == "__main__":
    main()
