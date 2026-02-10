"""
训练与评估模块 (优化版)
========================
改进点:
  1. 学习率调度器 (ReduceLROnPlateau): 验证集无提升时自动降低LR
  2. 权重衰减 (weight_decay): L2正则化防止过拟合
  3. 类别权重 (pos_weight): 处理 Fake:Real ≈ 60:40 的不平衡
  4. 梯度裁剪: 防止梯度爆炸
  5. Focal Loss: 聚焦困难样本
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, confusion_matrix)


# ========================= Focal Loss =========================
class FocalLoss(nn.Module):
    """
    Focal Loss: 解决类别不平衡 + 聚焦困难样本
    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        bce_loss = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction='none'
        )
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma
        loss = self.alpha * focal_weight * bce_loss
        return loss.mean()


# ========================= 创建带类别权重的损失函数 =========================
def create_weighted_bce_loss(train_labels):
    """
    根据训练集标签分布创建带权重的BCE损失
    pos_weight = num_negative / num_positive
    这样少数类(Real)的损失权重更高
    """
    labels = np.array(train_labels)
    num_pos = (labels == 1).sum()
    num_neg = (labels == 0).sum()
    pos_weight = torch.tensor([num_neg / num_pos], dtype=torch.float32)
    print(f"  类别权重 pos_weight = {pos_weight.item():.2f} "
          f"(Fake:{num_neg}, Real:{num_pos})")
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight)


# ========================= 训练一个epoch =========================
def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(inputs)
        loss = criterion(logits, labels)
        loss.backward()
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        preds = (torch.sigmoid(logits) > 0.5).float()
        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    return avg_loss, accuracy


# ========================= 评估函数 =========================
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            logits = model(inputs)
            loss = criterion(logits, labels)

            total_loss += loss.item()
            preds = (torch.sigmoid(logits) > 0.5).float()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)

    metrics = {
        'loss': avg_loss,
        'accuracy': accuracy_score(all_labels, all_preds),
        'precision': precision_score(all_labels, all_preds, zero_division=0),
        'recall': recall_score(all_labels, all_preds, zero_division=0),
        'f1': f1_score(all_labels, all_preds, zero_division=0),
        'confusion_matrix': confusion_matrix(all_labels, all_preds)
    }

    return metrics, all_preds, all_labels


# ========================= 完整训练流程 (优化版) =========================
def train_model(model, train_dataset, val_dataset, test_dataset,
                criterion, optimizer, device,
                num_epochs=10, batch_size=32, use_scheduler=True):
    """
    完整训练流程 (优化版):
    - ★ 学习率调度器: 验证集F1连续3个epoch无提升时, LR减半
    - ★ 保存验证集F1最优的模型 (而非accuracy, F1对不平衡数据更合理)
    - 记录完整训练历史
    """
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # ★ 学习率调度器
    scheduler = None
    if use_scheduler:
        scheduler = ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=3,
            min_lr=1e-6
        )

    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'test_acc': []
    }

    best_val_acc = 0
    best_model_state = None
    patience_counter = 0
    early_stop_patience = 5  # 连续5个epoch val不提升就停

    for epoch in range(num_epochs):
        # --- 训练 ---
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )

        # --- 验证 ---
        val_metrics, _, _ = evaluate(model, val_loader, criterion, device)

        # --- 测试 ---
        test_metrics, _, _ = evaluate(model, test_loader, criterion, device)

        # --- 记录历史 ---
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['accuracy'])
        history['test_acc'].append(test_metrics['accuracy'])

        # 当前学习率
        current_lr = optimizer.param_groups[0]['lr']

        print(f"  Epoch [{epoch+1}/{num_epochs}] "
              f"Loss: {train_loss:.4f} | Train: {train_acc:.4f} | "
              f"Val: {val_metrics['accuracy']:.4f}(F1:{val_metrics['f1']:.4f}) | "
              f"Test: {test_metrics['accuracy']:.4f} | LR: {current_lr:.6f}")

        # ★ 调度器: 根据验证集accuracy调整学习率
        if scheduler is not None:
            scheduler.step(val_metrics['accuracy'])

        # ★ 保存最优模型 (基于验证集accuracy)
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        # ★ 早停: 连续N个epoch val accuracy不提升就停止
        if patience_counter >= early_stop_patience:
            print(f"  Early stopping at epoch {epoch+1} (val acc not improving)")
            break

    # 加载最优模型参数
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return history, model
