"""
数据增强模块
============
功能:
  1. 加载外部新闻数据集 (News_dataset: Fake.csv + True.csv)
  2. 多数据集合并与去重
  3. 文本数据增强 (EDA: Easy Data Augmentation)
     - 随机删词 (Random Deletion)
     - 随机换词位置 (Random Swap)

参考:
  Jason Wei & Kai Zou, "EDA: Easy Data Augmentation Techniques
  for Boosting Performance on Text Classification Tasks", 2019
"""

import os
import random
import numpy as np
import pandas as pd


# ========================= 加载外部新闻数据集 =========================
def load_news_dataset(dataset_dir):
    """
    加载 News_dataset 目录下的 Fake.csv 和 True.csv, 合并为统一格式

    参数:
        dataset_dir: 数据集目录路径 (包含 Fake.csv 和 True.csv)

    返回:
        pd.DataFrame: 包含 'text' 和 'label' 两列
            - Fake.csv → label=0
            - True.csv → label=1
    """
    fake_path = os.path.join(dataset_dir, "Fake.csv")
    true_path = os.path.join(dataset_dir, "True.csv")

    dfs = []

    if os.path.exists(fake_path):
        df_fake = pd.read_csv(fake_path)
        print(f"    外部数据集 Fake.csv: {len(df_fake)} 条")

        # 合并 title + text 作为完整文本 (如果title存在且不为空)
        if 'title' in df_fake.columns:
            df_fake['text'] = df_fake.apply(
                lambda row: f"{row['title']}. {row['text']}"
                if pd.notna(row.get('title')) and str(row.get('title', '')).strip()
                else str(row.get('text', '')),
                axis=1
            )

        df_fake['label'] = 0
        dfs.append(df_fake[['text', 'label']])
    else:
        print(f"    ⚠ 未找到 Fake.csv: {fake_path}")

    if os.path.exists(true_path):
        df_true = pd.read_csv(true_path)
        print(f"    外部数据集 True.csv: {len(df_true)} 条")

        if 'title' in df_true.columns:
            df_true['text'] = df_true.apply(
                lambda row: f"{row['title']}. {row['text']}"
                if pd.notna(row.get('title')) and str(row.get('title', '')).strip()
                else str(row.get('text', '')),
                axis=1
            )

        df_true['label'] = 1
        dfs.append(df_true[['text', 'label']])
    else:
        print(f"    ⚠ 未找到 True.csv: {true_path}")

    if dfs:
        df_extra = pd.concat(dfs, ignore_index=True)
        print(f"    外部数据集合计: {len(df_extra)} 条 "
              f"(Fake: {(df_extra['label']==0).sum()}, "
              f"Real: {(df_extra['label']==1).sum()})")
        return df_extra
    else:
        print("    ⚠ 外部数据集为空")
        return pd.DataFrame(columns=['text', 'label'])


# ========================= 多数据集合并 =========================
def merge_datasets(df_list, dedup=True):
    """
    合并多个DataFrame, 可选去重

    参数:
        df_list: DataFrame列表, 每个需包含 'text' 和 'label' 列
        dedup: 是否基于文本前100字符去重

    返回:
        合并后的DataFrame
    """
    df = pd.concat(df_list, ignore_index=True)
    total_before = len(df)

    if dedup:
        # 基于文本前100字符去重 (避免完全相同的新闻)
        df['_dedup_key'] = df['text'].astype(str).str[:100]
        df = df.drop_duplicates(subset='_dedup_key', keep='first')
        df = df.drop(columns='_dedup_key')
        df = df.reset_index(drop=True)

    total_after = len(df)
    removed = total_before - total_after
    print(f"  合并数据集: {total_before} 条 → 去重后 {total_after} 条 "
          f"(移除 {removed} 条重复)")
    print(f"  合并后标签分布: Fake={int((df['label']==0).sum())}, "
          f"Real={int((df['label']==1).sum())}")

    return df


# ========================= 文本数据增强 (EDA) =========================
def _random_deletion(words, p=0.1):
    """
    随机删词: 以概率p删除每个词

    参数:
        words: 词列表
        p: 每个词被删除的概率

    返回:
        删除后的词列表 (至少保留1个词)
    """
    if len(words) <= 1:
        return words

    new_words = [w for w in words if random.random() > p]

    # 至少保留1个词
    if len(new_words) == 0:
        return [random.choice(words)]

    return new_words


def _random_swap(words, n=1):
    """
    随机换词: 随机交换n对词的位置

    参数:
        words: 词列表
        n: 交换次数

    返回:
        交换后的词列表
    """
    if len(words) < 2:
        return words

    new_words = words.copy()
    for _ in range(n):
        idx1, idx2 = random.sample(range(len(new_words)), 2)
        new_words[idx1], new_words[idx2] = new_words[idx2], new_words[idx1]

    return new_words


def augment_text(text, num_aug=1):
    """
    对单条文本进行数据增强

    参数:
        text: 已清洗的文本字符串
        num_aug: 生成的增强样本数

    返回:
        增强后的文本列表
    """
    words = text.split()
    if len(words) < 3:
        return [text] * num_aug  # 太短的文本不增强

    augmented = []
    for _ in range(num_aug):
        method = random.choice(['delete', 'swap'])

        if method == 'delete':
            new_words = _random_deletion(words, p=0.1)
        else:  # swap
            n_swaps = max(1, len(words) // 20)  # 每20个词换1对
            new_words = _random_swap(words, n=n_swaps)

        augmented.append(' '.join(new_words))

    return augmented


def augment_dataset(X_train, y_train, num_aug=1, seed=42):
    """
    对训练集进行文本数据增强

    参数:
        X_train: 训练集文本数组
        y_train: 训练集标签数组
        num_aug: 每条文本生成的增强样本数
        seed: 随机种子

    返回:
        增强后的 (X_train, y_train) — numpy数组
    """
    random.seed(seed)
    np.random.seed(seed)

    X_list = list(X_train)
    y_list = list(y_train)

    orig_size = len(X_list)

    for i in range(orig_size):
        text = X_list[i]
        label = y_list[i]

        aug_texts = augment_text(text, num_aug=num_aug)
        for aug_text in aug_texts:
            X_list.append(aug_text)
            y_list.append(label)

    # 打乱顺序
    combined = list(zip(X_list, y_list))
    random.shuffle(combined)
    X_aug, y_aug = zip(*combined)

    print(f"    数据增强: {orig_size} → {len(X_aug)} "
          f"(+{len(X_aug) - orig_size} 条增强样本)")

    return np.array(X_aug), np.array(y_aug)
