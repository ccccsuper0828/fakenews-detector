"""
Unified training and evaluation runner.

Provides a common train/eval loop that works with any model
returning logits from forward(input_ids) or forward(**kwargs).
"""

import os
import time
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from experiments.config import ExperimentConfig, TrainConfig
from experiments.metrics import (
    compute_metrics, count_parameters, measure_inference_time,
    EvalResult, ResultLogger,
)


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        weight = (1 - p_t) ** self.gamma
        return (self.alpha * weight * bce).mean()


def build_criterion(cfg: TrainConfig, device: str) -> nn.Module:
    if cfg.loss_fn == "focal":
        return FocalLoss(alpha=cfg.focal_alpha, gamma=cfg.focal_gamma).to(device)
    return nn.BCEWithLogitsLoss().to(device)


def train_one_epoch(model, dataloader, optimizer, criterion, device, grad_clip=1.0):
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for batch in dataloader:
        inputs, labels = batch[0].to(device), batch[1].to(device)
        optimizer.zero_grad()
        logits = model(inputs)
        if logits.dim() > 1:
            logits = logits.squeeze(-1)
        loss = criterion(logits, labels)
        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item()
        preds = (torch.sigmoid(logits) > 0.5).float()
        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())

    avg_loss = total_loss / max(len(dataloader), 1)
    acc = np.mean(np.array(all_preds) == np.array(all_labels))
    return avg_loss, acc


@torch.no_grad()
def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels, all_probs = [], [], []

    for batch in dataloader:
        inputs, labels = batch[0].to(device), batch[1].to(device)
        logits = model(inputs)
        if logits.dim() > 1:
            logits = logits.squeeze(-1)
        loss = criterion(logits, labels)

        total_loss += loss.item()
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    avg_loss = total_loss / max(len(dataloader), 1)
    return avg_loss, np.array(all_labels), np.array(all_preds), np.array(all_probs)


def run_training(
    model: nn.Module,
    train_dataset,
    val_dataset,
    test_dataset,
    cfg: ExperimentConfig,
    optimizer=None,
    scheduler=None,
) -> tuple:
    """
    Full training loop with early stopping and best model checkpointing.

    Returns:
        (history dict, best model, EvalResult on test set)
    """
    tc = cfg.train
    device = tc.resolve_device()
    model = model.to(device)

    train_loader = DataLoader(train_dataset, batch_size=tc.batch_size,
                              shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=tc.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=tc.batch_size, shuffle=False)

    criterion = build_criterion(tc, device)

    if optimizer is None:
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=tc.learning_rate,
            weight_decay=tc.weight_decay,
        )

    if scheduler is None and tc.use_scheduler:
        scheduler = ReduceLROnPlateau(
            optimizer, mode="max", factor=tc.scheduler_factor,
            patience=tc.scheduler_patience, min_lr=1e-7,
        )

    history = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [], "val_f1": [],
        "test_acc": [],
    }

    best_val_f1 = 0.0
    best_state = None
    patience_counter = 0

    for epoch in range(tc.num_epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device, tc.grad_clip_norm
        )

        val_loss, val_labels, val_preds, val_probs = evaluate_model(
            model, val_loader, criterion, device
        )
        val_acc = np.mean(val_labels == val_preds)
        val_f1 = float(np.mean(val_preds == val_labels))  # simplified; use sklearn below
        from sklearn.metrics import f1_score as sk_f1
        val_f1 = sk_f1(val_labels, val_preds, zero_division=0)

        _, test_labels, test_preds, _ = evaluate_model(
            model, test_loader, criterion, device
        )
        test_acc = np.mean(test_labels == test_preds)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)
        history["test_acc"].append(test_acc)

        lr_now = optimizer.param_groups[0]["lr"]
        print(
            f"  Epoch [{epoch+1}/{tc.num_epochs}] "
            f"Loss={train_loss:.4f} Train={train_acc:.4f} "
            f"Val={val_acc:.4f}(F1={val_f1:.4f}) "
            f"Test={test_acc:.4f} LR={lr_now:.6f}"
        )

        if scheduler is not None:
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

    # Final test evaluation
    _, test_labels, test_preds, test_probs = evaluate_model(
        model, test_loader, criterion, device
    )

    n_params = count_parameters(model)
    infer_ms = measure_inference_time(model, test_loader, device)

    result = compute_metrics(
        test_labels, test_preds, test_probs,
        experiment_name=cfg.name,
        group=cfg.group,
        param_count=n_params,
        inference_time_ms=infer_ms,
    )

    print(f"\n  Final: {result.summary()}")
    return history, model, result
