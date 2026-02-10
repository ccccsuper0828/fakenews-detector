"""
可视化模块
==========
满足作业要求的全部图表:
  Fig 1: BCE Loss - 训练损失/训练准确率/测试准确率 vs Epoch
  Fig 2: Focal Loss - 同上 (不同损失函数)
  Fig 3: BCE Loss - 不同学习率对比
  Fig 4: Focal Loss - 不同学习率对比
  Fig 5: BCE Loss - 不同batch size对比
  Fig 6: Focal Loss - 不同batch size对比
  Fig 7: 前100条测试样本的预测结果
  Fig 8: 混淆矩阵
"""

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix

# 避免中文乱码 (macOS)
matplotlib.rcParams['font.family'] = ['Arial', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False


# ========================= 单次实验训练曲线 =========================
def plot_training_curves(history, title="Training Curves", save_path=None):
    """
    绘制训练曲线: 训练损失 + 训练准确率 + 测试准确率
    作业要求: [one figure] 包含三条曲线
    """
    epochs = range(1, len(history['train_loss']) + 1)

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # 左轴: Loss
    color_loss = 'tab:red'
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', color=color_loss, fontsize=12)
    line1 = ax1.plot(epochs, history['train_loss'], 'r-o', label='Train Loss',
                     markersize=4, linewidth=2)
    ax1.tick_params(axis='y', labelcolor=color_loss)
    ax1.grid(True, alpha=0.3)

    # 右轴: Accuracy
    ax2 = ax1.twinx()
    color_acc = 'tab:blue'
    ax2.set_ylabel('Accuracy', color=color_acc, fontsize=12)
    line2 = ax2.plot(epochs, history['train_acc'], 'b-s', label='Train Accuracy',
                     markersize=4, linewidth=2)
    line3 = ax2.plot(epochs, history['test_acc'], 'g-^', label='Test Accuracy',
                     markersize=4, linewidth=2)
    ax2.tick_params(axis='y', labelcolor=color_acc)
    ax2.set_ylim([0, 1.05])

    # 合并图例
    lines = line1 + line2 + line3
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='center right', fontsize=10)

    plt.title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"    图表已保存: {save_path}")
    plt.close()


# ========================= 学习率对比图 =========================
def plot_lr_comparison(histories_dict, title="Learning Rate Comparison", save_path=None):
    """
    绘制不同学习率的对比图
    作业要求: [one figure] 包含三个子图 (Loss / Train Acc / Test Acc)
    """
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(histories_dict)))

    for idx, (lr, history) in enumerate(histories_dict.items()):
        epochs = range(1, len(history['train_loss']) + 1)
        c = colors[idx]

        axes[0].plot(epochs, history['train_loss'], color=c, marker='o',
                     markersize=3, linewidth=1.5, label=f'LR={lr}')
        axes[1].plot(epochs, history['train_acc'], color=c, marker='s',
                     markersize=3, linewidth=1.5, label=f'LR={lr}')
        axes[2].plot(epochs, history['test_acc'], color=c, marker='^',
                     markersize=3, linewidth=1.5, label=f'LR={lr}')

    for ax, ylabel, subtitle in zip(
        axes,
        ['Loss', 'Accuracy', 'Accuracy'],
        ['Training Loss', 'Training Accuracy', 'Test Accuracy']
    ):
        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(subtitle, fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"    图表已保存: {save_path}")
    plt.close()


# ========================= Batch Size对比图 =========================
def plot_batch_size_comparison(histories_dict, title="Batch Size Comparison", save_path=None):
    """
    绘制不同batch size的对比图
    格式与学习率对比图相同
    """
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(histories_dict)))

    for idx, (bs, history) in enumerate(histories_dict.items()):
        epochs = range(1, len(history['train_loss']) + 1)
        c = colors[idx]

        axes[0].plot(epochs, history['train_loss'], color=c, marker='o',
                     markersize=3, linewidth=1.5, label=f'BS={bs}')
        axes[1].plot(epochs, history['train_acc'], color=c, marker='s',
                     markersize=3, linewidth=1.5, label=f'BS={bs}')
        axes[2].plot(epochs, history['test_acc'], color=c, marker='^',
                     markersize=3, linewidth=1.5, label=f'BS={bs}')

    for ax, ylabel, subtitle in zip(
        axes,
        ['Loss', 'Accuracy', 'Accuracy'],
        ['Training Loss', 'Training Accuracy', 'Test Accuracy']
    ):
        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(subtitle, fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"    图表已保存: {save_path}")
    plt.close()


# ========================= 预测结果可视化 =========================
def plot_predictions_table(texts, true_labels, pred_labels, n=100, save_path=None):
    """
    可视化前n条测试集预测结果
    作业要求: [one figure or one table]
    使用网格色块直观展示正确(绿)/错误(红)
    """
    n = min(n, len(texts))
    label_map = {0: 'Fake', 1: 'Real'}

    # 方式1: 网格概览图 (10×10)
    fig, axes = plt.subplots(1, 2, figsize=(16, 8),
                             gridspec_kw={'width_ratios': [1, 3]})

    # 左: 预测正确性网格
    grid_size = int(np.ceil(np.sqrt(n)))
    grid = np.zeros((grid_size, grid_size))
    for i in range(n):
        row, col = i // grid_size, i % grid_size
        grid[row][col] = 1 if true_labels[i] == pred_labels[i] else -1

    # 自定义颜色: 绿色=正确, 红色=错误, 白色=空
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(['#ff6b6b', 'white', '#51cf66'])
    im = axes[0].imshow(grid, cmap=cmap, vmin=-1, vmax=1, aspect='equal')
    axes[0].set_title(f'Prediction Grid (First {n} samples)\nGreen=Correct, Red=Wrong',
                      fontsize=11)
    axes[0].set_xlabel('Column')
    axes[0].set_ylabel('Row')

    # 右: 文本表格 (前20条详情)
    axes[1].axis('off')
    show_n = min(20, n)
    table_data = []
    cell_colors = []
    for i in range(show_n):
        text_preview = texts[i][:60] + '...' if len(texts[i]) > 60 else texts[i]
        true_l = label_map[int(true_labels[i])]
        pred_l = label_map[int(pred_labels[i])]
        correct = 'Yes' if true_labels[i] == pred_labels[i] else 'No'
        table_data.append([i+1, text_preview, true_l, pred_l, correct])

        if true_labels[i] == pred_labels[i]:
            cell_colors.append(['#d4edda'] * 5)  # 浅绿
        else:
            cell_colors.append(['#f8d7da'] * 5)  # 浅红

    table = axes[1].table(
        cellText=table_data,
        colLabels=['#', 'Text (preview)', 'True', 'Pred', 'Correct'],
        cellColours=cell_colors,
        loc='upper center',
        cellLoc='left'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.auto_set_column_width([0, 1, 2, 3, 4])
    axes[1].set_title(f'Detailed Predictions (First {show_n} samples)', fontsize=11)

    # 统计信息
    correct_count = sum(1 for i in range(n) if true_labels[i] == pred_labels[i])
    fig.suptitle(
        f'Test Set Predictions: {correct_count}/{n} correct ({100*correct_count/n:.1f}%)',
        fontsize=14, fontweight='bold', y=1.02
    )

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"    图表已保存: {save_path}")
    plt.close()


# ========================= 混淆矩阵 =========================
def plot_confusion_matrix(true_labels, pred_labels, save_path=None):
    """绘制混淆矩阵热力图"""
    cm = confusion_matrix(true_labels, pred_labels)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Fake (0)', 'Real (1)'],
                yticklabels=['Fake (0)', 'Real (1)'],
                ax=ax, annot_kws={"size": 16})
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"    图表已保存: {save_path}")
    plt.close()
