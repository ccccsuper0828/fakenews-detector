"""
Group 2 Runner: LLM fine-tuning experiments.

Usage:
    python -m experiments.group2_llm.run_group2 [--quick]
"""

import os
import sys
import argparse
import re

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from experiments.config import ExperimentConfig, TransformerConfig, TrainConfig
from experiments.metrics import (
    compute_metrics, count_parameters, ResultLogger, plot_comparison_table, EvalResult,
)
from experiments.runner import FocalLoss, build_criterion
from experiments.group2_llm.models import TransformerClassifier, TransformerDataset


def clean_text_bert(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def load_data_for_transformer(cfg: ExperimentConfig, tokenizer):
    """Load and split data, returning TransformerDatasets."""
    base = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    dc = cfg.data

    csv_path = os.path.join(base, dc.csv_path)
    df_main = pd.read_csv(csv_path)

    extra_dir = os.path.join(base, dc.extra_dataset_dir) if dc.extra_dataset_dir else None
    if extra_dir and os.path.isdir(extra_dir):
        from src.data_augment import load_news_dataset, merge_datasets
        df_extra = load_news_dataset(extra_dir)
        df = merge_datasets([df_main, df_extra], dedup=True)
    else:
        df = df_main

    df = df.dropna(subset=["text", "label"])
    df["clean_text"] = df["text"].apply(clean_text_bert)
    df = df[df["clean_text"].str.len() > 0].reset_index(drop=True)

    texts = df["clean_text"].values
    labels = df["label"].values.astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=dc.test_size,
        random_state=dc.random_state, stratify=labels,
    )
    val_ratio = dc.val_size / (1 - dc.test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=val_ratio,
        random_state=dc.random_state, stratify=y_train,
    )

    max_len = cfg.transformer.max_seq_length
    train_ds = TransformerDataset(X_train, y_train, tokenizer, max_len)
    val_ds = TransformerDataset(X_val, y_val, tokenizer, max_len)
    test_ds = TransformerDataset(X_test, y_test, tokenizer, max_len)

    print(f"  Data loaded: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")
    return train_ds, val_ds, test_ds


def train_transformer(model, train_ds, val_ds, test_ds, cfg: ExperimentConfig):
    """Training loop specialized for transformer models (3-element batches)."""
    tc = cfg.train
    device = tc.resolve_device()
    model = model.to(device)

    train_loader = DataLoader(train_ds, batch_size=tc.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=tc.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=tc.batch_size, shuffle=False)

    criterion = build_criterion(tc, device)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=tc.learning_rate, weight_decay=tc.weight_decay,
    )
    scheduler = ReduceLROnPlateau(
        optimizer, mode="max", factor=tc.scheduler_factor,
        patience=tc.scheduler_patience, min_lr=1e-7,
    )

    best_val_f1 = 0.0
    best_state = None
    patience_counter = 0

    for epoch in range(tc.num_epochs):
        model.train()
        total_loss = 0.0
        all_preds, all_labels = [], []

        for input_ids, attn_mask, labels in train_loader:
            input_ids = input_ids.to(device)
            attn_mask = attn_mask.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(input_ids, attention_mask=attn_mask)
            loss = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip_norm)
            optimizer.step()

            total_loss += loss.item()
            preds = (torch.sigmoid(logits) > 0.5).float()
            all_preds.extend(preds.detach().cpu().numpy())
            all_labels.extend(labels.detach().cpu().numpy())

        train_loss = total_loss / max(len(train_loader), 1)
        train_acc = np.mean(np.array(all_preds) == np.array(all_labels))

        val_loss, val_labels, val_preds, val_probs = _eval_transformer(
            model, val_loader, criterion, device
        )
        from sklearn.metrics import f1_score as sk_f1
        val_f1 = sk_f1(val_labels, val_preds, zero_division=0)
        val_acc = np.mean(val_labels == val_preds)

        lr_now = optimizer.param_groups[0]["lr"]
        print(
            f"  Epoch [{epoch+1}/{tc.num_epochs}] "
            f"Loss={train_loss:.4f} Train={train_acc:.4f} "
            f"Val={val_acc:.4f}(F1={val_f1:.4f}) LR={lr_now:.6f}"
        )

        scheduler.step(val_f1)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= tc.early_stop_patience:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    _, test_labels, test_preds, test_probs = _eval_transformer(
        model, test_loader, criterion, device
    )

    n_params = count_parameters(model)
    result = compute_metrics(
        test_labels, test_preds, test_probs,
        experiment_name=cfg.name, group=cfg.group,
        param_count=n_params,
    )
    print(f"\n  Final: {result.summary()}")
    return model, result


@torch.no_grad()
def _eval_transformer(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels, all_probs = [], [], []

    for input_ids, attn_mask, labels in dataloader:
        input_ids = input_ids.to(device)
        attn_mask = attn_mask.to(device)
        labels = labels.to(device)

        logits = model(input_ids, attention_mask=attn_mask)
        loss = criterion(logits, labels)
        total_loss += loss.item()

        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    avg_loss = total_loss / max(len(dataloader), 1)
    return avg_loss, np.array(all_labels), np.array(all_preds), np.array(all_probs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print("  Group 2: LLM Fine-tuning Experiments")
    print("=" * 60)

    output_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "outputs", "group2_llm",
    )
    os.makedirs(output_dir, exist_ok=True)
    logger = ResultLogger(os.path.join(output_dir, "results.csv"))

    model_configs = [
        ("distilbert-base-uncased", "distilbert"),
        ("bert-base-uncased", "bert"),
        ("roberta-base", "roberta"),
    ]
    if args.quick:
        model_configs = [("distilbert-base-uncased", "distilbert")]

    freeze_strategies = [
        ("none", 0),
        ("all", 0),
        ("top_n", 4),
    ]
    if args.quick:
        freeze_strategies = [("none", 0)]

    all_results = []

    for model_name, short_name in model_configs:
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        for freeze_strat, top_n in freeze_strategies:
            exp_name = f"G2_{short_name}_{freeze_strat}"
            if freeze_strat == "top_n":
                exp_name += f"{top_n}"

            print(f"\n{'='*60}\n  Experiment: {exp_name}\n{'='*60}")

            cfg = ExperimentConfig(
                name=exp_name, group="group2_llm",
                train=TrainConfig(
                    num_epochs=3 if args.quick else 10,
                    batch_size=16,
                    learning_rate=2e-5 if freeze_strat == "none" else 1e-3,
                    early_stop_patience=3,
                ),
                transformer=TransformerConfig(
                    model_name=model_name,
                    freeze_strategy=freeze_strat,
                    unfreeze_top_n=top_n,
                    max_seq_length=256,
                ),
            )

            train_ds, val_ds, test_ds = load_data_for_transformer(cfg, tokenizer)

            model = TransformerClassifier(
                model_name=model_name,
                freeze_strategy=freeze_strat,
                unfreeze_top_n=top_n,
                classifier_hidden=cfg.transformer.classifier_hidden,
                classifier_dropout=cfg.transformer.classifier_dropout,
            )

            _, result = train_transformer(model, train_ds, val_ds, test_ds, cfg)
            all_results.append(result)
            logger.log(result)

    plot_comparison_table(
        all_results,
        save_path=os.path.join(output_dir, "group2_comparison.png"),
        title="Group 2: Transformer Model Comparison",
    )

    print(f"\n  All Group 2 results saved to {output_dir}")


if __name__ == "__main__":
    main()
