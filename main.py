"""
============================================================
  虚假新闻检测 - 基于深度学习的完整项目 (GloVe优化版)
  Fake News Detection with Deep Learning
  CDS525 Group Project
============================================================

项目结构:
  main.py                    ← 主入口 (当前文件)
  src/
    data_utils.py            ← 数据加载与预处理
    model.py                 ← BiLSTM + Attention 分类模型
    trainer.py               ← 训练/评估/损失函数
    visualize.py             ← 可视化 (满足作业全部图表要求)
    chain_of_thought.py      ← 推理链模块 (可解释性)

运行方式:
  python main.py

优化措施:
  ★ 合并数据集 — 原始数据(5K) + News_dataset(45K) ≈ 50K训练数据
  ★ 文本数据增强 (EDA) — 随机删词/换词, 进一步扩充训练集
  ★ GloVe预训练词向量 — 97%词汇覆盖率, 大幅提升语义表示
  ★ 冻结GloVe — 减少2M可训练参数, 有效抑制过拟合
  ★ AdamW + weight_decay — L2正则化
  ★ ReduceLROnPlateau — 验证集不提升时自动降低学习率
  ★ Early Stopping — 防止过拟合训练过长
  ★ Val Accuracy模型选择 — 比Val F1更稳定
  ★ 去停用词 + 智能截断 (70%头+30%尾)
"""

import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader

from src.data_utils import (load_and_preprocess_data, load_and_preprocess_multi_data,
                            load_glove_embeddings)
from src.model import BiLSTMAttentionClassifier
from src.trainer import (train_model, evaluate, FocalLoss)
from src.visualize import (plot_training_curves, plot_lr_comparison,
                           plot_batch_size_comparison, plot_predictions_table,
                           plot_confusion_matrix)
from src.chain_of_thought import ChainOfThoughtAnalyzer, batch_analyze


# ========================= 全局配置 =========================
DATA_PATH = os.path.join(os.path.dirname(__file__), "fakenews 2.csv")
EXTRA_DATA_DIR = os.path.join(os.path.dirname(__file__), "News _dataset")
GLOVE_PATH = os.path.join(os.path.dirname(__file__), "glove.6B.100d.txt")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 默认超参数 (GloVe优化版)
CONFIG = {
    'embed_dim': 100,        # ★ GloVe维度 (必须匹配glove.6B.100d.txt)
    'hidden_dim': 128,       # LSTM隐藏层维度
    'num_layers': 2,         # LSTM层数
    'dropout': 0.5,          # Dropout比率
    'max_vocab_size': 20000, # 词汇表大小
    'max_length': 500,       # 文本最大长度 (token数)
    'num_epochs': 20,        # ★ 增加epoch数 (冻结GloVe收敛更慢但更稳定)
    'batch_size': 32,        # 批大小
    'learning_rate': 0.001,  # 学习率
    'weight_decay': 1e-4,    # ★ 权重衰减 (L2正则化)
    'freeze_embeddings': True,  # ★ 冻结GloVe嵌入层
}

# 全局变量 (避免重复加载)
_glove_matrix = None
_vocab = None


def create_model(vocab_size, glove_matrix=None):
    """创建新的模型实例 (带GloVe)"""
    model = BiLSTMAttentionClassifier(
        vocab_size=vocab_size,
        embed_dim=CONFIG['embed_dim'],
        hidden_dim=CONFIG['hidden_dim'],
        num_layers=CONFIG['num_layers'],
        dropout=CONFIG['dropout'],
        pretrained_embeddings=glove_matrix,
        freeze_embeddings=CONFIG['freeze_embeddings']
    )
    return model.to(DEVICE)


def run_experiment(train_ds, val_ds, test_ds, vocab_size,
                   loss_fn='bce', learning_rate=None, batch_size=None,
                   num_epochs=None, use_scheduler=True, glove_matrix=None):
    """
    运行单次训练实验

    参数:
        loss_fn: 'bce' 或 'focal' (FocalLoss)
        learning_rate: 学习率 (None则使用默认值)
        batch_size: 批大小 (None则使用默认值)
        num_epochs: 训练轮数 (None则使用默认值)
        use_scheduler: 是否使用学习率调度器
        glove_matrix: GloVe embedding矩阵

    返回:
        history: 训练历史
        model: 训练好的模型
    """
    lr = learning_rate or CONFIG['learning_rate']
    bs = batch_size or CONFIG['batch_size']
    epochs = num_epochs or CONFIG['num_epochs']

    model = create_model(vocab_size, glove_matrix)

    if loss_fn == 'bce':
        criterion = nn.BCEWithLogitsLoss()
    elif loss_fn == 'focal':
        criterion = FocalLoss(alpha=0.25, gamma=2.0)
    else:
        raise ValueError(f"未知损失函数: {loss_fn}")

    criterion = criterion.to(DEVICE)

    # ★ AdamW (只优化可训练参数, 冻结的GloVe不参与)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=CONFIG['weight_decay']
    )

    history, best_model = train_model(
        model, train_ds, val_ds, test_ds,
        criterion, optimizer, DEVICE,
        num_epochs=epochs, batch_size=bs,
        use_scheduler=use_scheduler
    )

    return history, best_model


def main():
    global _glove_matrix, _vocab

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("  Fake News Detection with Deep Learning")
    print("  BiLSTM + Attention + GloVe + Chain-of-Thought")
    print(f"  Device: {DEVICE}")
    print("=" * 60)

    # ============================================================
    # Step 1: 数据加载与预处理 + GloVe
    # ============================================================
    print("\n" + "=" * 60)
    print("[Step 1/7] Loading data and GloVe embeddings...")
    print("=" * 60)

    train_ds, val_ds, test_ds, vocab = load_and_preprocess_multi_data(
        DATA_PATH,
        extra_dataset_dir=EXTRA_DATA_DIR,
        max_vocab_size=CONFIG['max_vocab_size'],
        max_length=CONFIG['max_length'],
        augment=True,        # ★ 开启文本数据增强
        num_aug=1,           # 每条训练样本生成1条增强样本
    )
    _vocab = vocab

    vocab_size = vocab.vocab_size

    # ★ 加载GloVe预训练词向量
    glove_matrix = None
    if os.path.exists(GLOVE_PATH):
        glove_matrix, coverage = load_glove_embeddings(
            GLOVE_PATH, vocab, embed_dim=CONFIG['embed_dim']
        )
        _glove_matrix = glove_matrix
        print(f"  ★ GloVe loaded! Coverage: {coverage:.2%}")
    else:
        print(f"  ⚠ GloVe file not found: {GLOVE_PATH}")
        print(f"    Will use random embeddings instead.")

    print(f"\n  Data loaded successfully!")
    print(f"  Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
    print(f"  Vocabulary: {vocab_size} words")

    # ============================================================
    # Step 2: 实验1 - BCE Loss 默认配置 (Fig 1)
    # ============================================================
    print("\n" + "=" * 60)
    print("[Step 2/7] Experiment 1: BCE Loss (default config)")
    print("=" * 60)

    history_bce, model_bce = run_experiment(
        train_ds, val_ds, test_ds, vocab_size,
        loss_fn='bce', glove_matrix=glove_matrix
    )
    plot_training_curves(
        history_bce,
        title="Fig 1: BCE Loss - Default Config (GloVe)",
        save_path=os.path.join(OUTPUT_DIR, "fig1_bce_default.png")
    )

    # ============================================================
    # Step 3: 实验2 - Focal Loss 默认配置 (Fig 2)
    # ============================================================
    print("\n" + "=" * 60)
    print("[Step 3/7] Experiment 2: Focal Loss (default config)")
    print("=" * 60)

    history_focal, model_focal = run_experiment(
        train_ds, val_ds, test_ds, vocab_size,
        loss_fn='focal', glove_matrix=glove_matrix
    )
    plot_training_curves(
        history_focal,
        title="Fig 2: Focal Loss - Default Config (GloVe)",
        save_path=os.path.join(OUTPUT_DIR, "fig2_focal_default.png")
    )

    # ============================================================
    # Step 4: 实验3 - 不同学习率对比 (Fig 3 & Fig 4)
    # ============================================================
    print("\n" + "=" * 60)
    print("[Step 4/7] Experiment 3: Learning Rate Comparison")
    print("=" * 60)

    learning_rates = [0.01, 0.001, 0.0001, 0.00001]

    # Fig 3: BCE + 不同LR
    print("\n  --- BCE Loss ---")
    lr_histories_bce = {}
    for lr in learning_rates:
        print(f"\n  LR = {lr}:")
        h, _ = run_experiment(train_ds, val_ds, test_ds, vocab_size,
                              loss_fn='bce', learning_rate=lr,
                              glove_matrix=glove_matrix)
        lr_histories_bce[lr] = h

    plot_lr_comparison(
        lr_histories_bce,
        title="Fig 3: BCE Loss - Learning Rate Comparison",
        save_path=os.path.join(OUTPUT_DIR, "fig3_lr_bce.png")
    )

    # Fig 4: Focal + 不同LR
    print("\n  --- Focal Loss ---")
    lr_histories_focal = {}
    for lr in learning_rates:
        print(f"\n  LR = {lr}:")
        h, _ = run_experiment(train_ds, val_ds, test_ds, vocab_size,
                              loss_fn='focal', learning_rate=lr,
                              glove_matrix=glove_matrix)
        lr_histories_focal[lr] = h

    plot_lr_comparison(
        lr_histories_focal,
        title="Fig 4: Focal Loss - Learning Rate Comparison",
        save_path=os.path.join(OUTPUT_DIR, "fig4_lr_focal.png")
    )

    # ============================================================
    # Step 5: 实验4 - 不同Batch Size对比 (Fig 5 & Fig 6)
    # ============================================================
    print("\n" + "=" * 60)
    print("[Step 5/7] Experiment 4: Batch Size Comparison")
    print("=" * 60)

    batch_sizes = [16, 32, 64, 128]

    # Fig 5: BCE + 不同BS
    print("\n  --- BCE Loss ---")
    bs_histories_bce = {}
    for bs in batch_sizes:
        print(f"\n  Batch Size = {bs}:")
        h, _ = run_experiment(train_ds, val_ds, test_ds, vocab_size,
                              loss_fn='bce', batch_size=bs,
                              glove_matrix=glove_matrix)
        bs_histories_bce[bs] = h

    plot_batch_size_comparison(
        bs_histories_bce,
        title="Fig 5: BCE Loss - Batch Size Comparison",
        save_path=os.path.join(OUTPUT_DIR, "fig5_bs_bce.png")
    )

    # Fig 6: Focal + 不同BS
    print("\n  --- Focal Loss ---")
    bs_histories_focal = {}
    for bs in batch_sizes:
        print(f"\n  Batch Size = {bs}:")
        h, _ = run_experiment(train_ds, val_ds, test_ds, vocab_size,
                              loss_fn='focal', batch_size=bs,
                              glove_matrix=glove_matrix)
        bs_histories_focal[bs] = h

    plot_batch_size_comparison(
        bs_histories_focal,
        title="Fig 6: Focal Loss - Batch Size Comparison",
        save_path=os.path.join(OUTPUT_DIR, "fig6_bs_focal.png")
    )

    # ============================================================
    # Step 6: 预测结果可视化 (Fig 7 & Fig 8)
    # ============================================================
    print("\n" + "=" * 60)
    print("[Step 6/7] Generating prediction visualizations...")
    print("=" * 60)

    # 使用BCE模型的最优结果进行预测
    criterion_eval = nn.BCEWithLogitsLoss()
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)
    test_metrics, all_preds, all_labels = evaluate(
        model_bce, test_loader, criterion_eval, DEVICE
    )

    print(f"\n  === Final Test Results (Best BCE + GloVe Model) ===")
    print(f"  Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  Recall:    {test_metrics['recall']:.4f}")
    print(f"  F1 Score:  {test_metrics['f1']:.4f}")

    # Fig 7: 前100条预测结果
    plot_predictions_table(
        list(test_ds.texts[:100]),
        all_labels[:100],
        all_preds[:100],
        n=100,
        save_path=os.path.join(OUTPUT_DIR, "fig7_predictions.png")
    )

    # Fig 8: 混淆矩阵
    plot_confusion_matrix(
        all_labels, all_preds,
        save_path=os.path.join(OUTPUT_DIR, "fig8_confusion_matrix.png")
    )

    # ============================================================
    # Step 7: 推理链 (Chain-of-Thought) Demo
    # ============================================================
    print("\n" + "=" * 60)
    print("[Step 7/7] Chain-of-Thought Reasoning Demo")
    print("=" * 60)

    cot_analyzer = ChainOfThoughtAnalyzer(model_bce, vocab, DEVICE, CONFIG['max_length'])

    # 对测试集前5条进行推理链分析
    batch_analyze(cot_analyzer, list(test_ds.texts), n=5)

    # ============================================================
    # 完成!
    # ============================================================
    print("\n" + "=" * 60)
    print("  All experiments completed!")
    print(f"  Output directory: {OUTPUT_DIR}")
    print(f"  Generated figures: fig1 ~ fig8")
    print("=" * 60)


if __name__ == '__main__':
    main()
