"""
数据加载与预处理模块
====================
功能:
  1. 加载CSV数据集 (支持单数据集 / 多数据集合并)
  2. 文本清洗 (去HTML/URL/特殊字符, 小写转换, 去高频停用词)
  3. 构建词汇表 (Vocabulary)
  4. 创建PyTorch Dataset, 划分训练/验证/测试集
  5. ★ 支持外部数据集 (News_dataset) 合并 + 文本数据增强
"""

import re
import pandas as pd
import numpy as np
from collections import Counter
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

# ========================= 停用词列表 =========================
# 只去除最高频的功能词 (保留情感词/程度词等有区分力的词)
STOP_WORDS = set([
    'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
    'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been',
    'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
    'could', 'should', 'may', 'might', 'shall', 'can', 'it', 'its',
    'that', 'which', 'who', 'whom', 'this', 'these', 'those', 'am',
    'i', 'me', 'my', 'we', 'our', 'you', 'your', 'he', 'him', 'his',
    'she', 'her', 'they', 'them', 'their', 'as', 'if', 'when', 'than',
    'so', 'no', 'not', 'up', 'out', 'about', 'into', 'over', 'after',
    'before', 'between', 'under', 'again', 'then', 'once', 'here',
    'there', 'where', 'how', 'all', 'both', 'each', 'more', 'other',
    'some', 'such', 'own', 'same', 'just', 'now', 's', 't', 'd', 'm',
])


# ========================= 文本清洗 =========================
def clean_text(text):
    """
    文本清洗函数:
    1. 转小写
    2. 去除HTML标签
    3. 去除URL链接
    4. 只保留英文字母和空格
    5. 去除高频功能性停用词 (保留情感/程度词)
    6. 去除多余空格
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'<.*?>', ' ', text)          # 去HTML标签
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)  # 去URL
    text = re.sub(r'[^a-z\s]', ' ', text)       # 只保留字母
    text = re.sub(r'\s+', ' ', text).strip()     # 去多余空格
    # 去除高频功能性停用词 (保留有区分力的词)
    words = [w for w in text.split() if w not in STOP_WORDS and len(w) > 1]
    return " ".join(words)


# ========================= 词汇表 =========================
class Vocabulary:
    """
    词汇表: 将文本单词映射为整数索引
    - <PAD> = 0: 填充符号
    - <UNK> = 1: 未知词符号
    """

    def __init__(self, max_vocab_size=20000):
        self.word2idx = {'<PAD>': 0, '<UNK>': 1}
        self.idx2word = {0: '<PAD>', 1: '<UNK>'}
        self.max_vocab_size = max_vocab_size

    def build(self, texts):
        """从文本列表中构建词汇表 (取最高频的max_vocab_size个词)"""
        counter = Counter()
        for text in texts:
            counter.update(text.split())

        for word, _ in counter.most_common(self.max_vocab_size - 2):
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word

    def encode(self, text, max_length):
        """
        将文本转为定长索引序列
        ★ 智能截断: 如果文本超长, 取前70%和后30%的token拼接
          这样既保留了开头(通常含关键信息)也保留了结尾(结论/总结)
        """
        tokens = text.split()
        if len(tokens) > max_length:
            # 智能截断: 前70% + 后30%
            head_len = int(max_length * 0.7)
            tail_len = max_length - head_len
            tokens = tokens[:head_len] + tokens[-tail_len:]

        indices = [self.word2idx.get(w, 1) for w in tokens]  # 1 = <UNK>
        # 填充到 max_length
        padding_len = max_length - len(indices)
        if padding_len > 0:
            indices = indices + [0] * padding_len
        return indices

    def decode(self, indices):
        """将索引序列转回文本"""
        words = [self.idx2word.get(idx, '<UNK>') for idx in indices if idx != 0]
        return ' '.join(words)

    @property
    def vocab_size(self):
        return len(self.word2idx)


# ========================= PyTorch Dataset =========================
class FakeNewsDataset(Dataset):
    """假新闻检测数据集"""

    def __init__(self, texts, labels, vocab, max_length=500):
        self.texts = texts          # 清洗后的文本列表
        self.labels = labels        # 标签列表 (0=fake, 1=real)
        self.vocab = vocab
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        indices = self.vocab.encode(text, self.max_length)
        return (torch.tensor(indices, dtype=torch.long),
                torch.tensor(label, dtype=torch.float))


# ========================= 数据加载主函数 =========================
def load_and_preprocess_data(csv_path, max_vocab_size=20000, max_length=500,
                             test_size=0.2, val_size=0.1, random_state=42):
    """
    加载数据 → 清洗 → 划分 → 构建词汇表 → 创建Dataset

    参数:
        csv_path: CSV文件路径 (需要包含 'text' 和 'label' 两列)
        max_vocab_size: 词汇表最大大小
        max_length: 文本最大长度 (token数)
        test_size: 测试集比例
        val_size: 验证集比例 (相对于全部数据)
        random_state: 随机种子

    返回:
        train_dataset, val_dataset, test_dataset, vocab
    """
    # 1. 加载数据
    print(f"  加载数据: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"  原始数据量: {len(df)}")
    print(f"  列名: {list(df.columns)}")
    print(f"  标签分布: {df['label'].value_counts().to_dict()}")

    # 2. 清洗
    df = df.dropna(subset=['text', 'label'])
    df['clean_text'] = df['text'].apply(clean_text)
    df = df[df['clean_text'].str.len() > 0].reset_index(drop=True)
    print(f"  清洗后数据量: {len(df)}")

    texts = df['clean_text'].values
    labels = df['label'].values.astype(int)

    # 3. 划分训练集/测试集
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels
    )

    # 4. 从训练集中划分验证集
    val_ratio = val_size / (1 - test_size)  # 验证集占训练集的比例
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train,
        test_size=val_ratio,
        random_state=random_state,
        stratify=y_train
    )

    print(f"  训练集: {len(X_train)}, 验证集: {len(X_val)}, 测试集: {len(X_test)}")

    # 5. 构建词汇表 (仅基于训练集)
    vocab = Vocabulary(max_vocab_size)
    vocab.build(X_train)
    print(f"  词汇表大小: {vocab.vocab_size}")

    # 6. 创建Dataset
    train_dataset = FakeNewsDataset(X_train, y_train, vocab, max_length)
    val_dataset = FakeNewsDataset(X_val, y_val, vocab, max_length)
    test_dataset = FakeNewsDataset(X_test, y_test, vocab, max_length)

    return train_dataset, val_dataset, test_dataset, vocab


# ========================= 多数据集加载主函数 =========================
def load_and_preprocess_multi_data(csv_path, extra_dataset_dir=None,
                                   max_vocab_size=20000, max_length=500,
                                   test_size=0.2, val_size=0.1,
                                   random_state=42, augment=False,
                                   num_aug=1):
    """
    加载多个数据集 → 合并 → 清洗 → 划分 → 构建词汇表 → (可选)增强 → 创建Dataset

    参数:
        csv_path: 主CSV文件路径 (需要包含 'text' 和 'label' 两列)
        extra_dataset_dir: 额外数据集目录 (如 News_dataset/, 包含 Fake.csv 和 True.csv)
        max_vocab_size: 词汇表最大大小
        max_length: 文本最大长度 (token数)
        test_size: 测试集比例
        val_size: 验证集比例 (相对于全部数据)
        random_state: 随机种子
        augment: 是否对训练集进行文本增强
        num_aug: 每条文本生成的增强样本数

    返回:
        train_dataset, val_dataset, test_dataset, vocab
    """
    from src.data_augment import load_news_dataset, merge_datasets, augment_dataset

    # 1. 加载主数据集
    print(f"  加载主数据集: {csv_path}")
    df_main = pd.read_csv(csv_path)
    print(f"    主数据集: {len(df_main)} 条")

    # 2. 加载外部数据集 (如有)
    df_list = [df_main]
    if extra_dataset_dir is not None:
        df_extra = load_news_dataset(extra_dataset_dir)
        df_list.append(df_extra)

    # 3. 合并 + 去重
    if len(df_list) > 1:
        df = merge_datasets(df_list, dedup=True)
    else:
        df = df_main

    # 4. 清洗
    df = df.dropna(subset=['text', 'label'])
    print(f"  开始文本清洗...")
    df['clean_text'] = df['text'].apply(clean_text)
    df = df[df['clean_text'].str.len() > 0].reset_index(drop=True)
    print(f"  清洗后数据量: {len(df)}")
    print(f"  标签分布: Fake={int((df['label']==0).sum())}, "
          f"Real={int((df['label']==1).sum())}")

    texts = df['clean_text'].values
    labels = df['label'].values.astype(int)

    # 5. 划分训练集/测试集
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels
    )

    # 6. 从训练集中划分验证集
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train,
        test_size=val_ratio,
        random_state=random_state,
        stratify=y_train
    )

    print(f"  训练集: {len(X_train)}, 验证集: {len(X_val)}, 测试集: {len(X_test)}")

    # 7. ★ 文本数据增强 (仅对训练集)
    if augment and num_aug > 0:
        print(f"  ★ 对训练集进行数据增强 (num_aug={num_aug})...")
        X_train, y_train = augment_dataset(
            X_train, y_train, num_aug=num_aug, seed=random_state
        )
        print(f"  增强后训练集: {len(X_train)}")

    # 8. 构建词汇表 (仅基于训练集)
    vocab = Vocabulary(max_vocab_size)
    vocab.build(X_train)
    print(f"  词汇表大小: {vocab.vocab_size}")

    # 9. 创建Dataset
    train_dataset = FakeNewsDataset(X_train, y_train, vocab, max_length)
    val_dataset = FakeNewsDataset(X_val, y_val, vocab, max_length)
    test_dataset = FakeNewsDataset(X_test, y_test, vocab, max_length)

    return train_dataset, val_dataset, test_dataset, vocab


# ========================= GloVe预训练词向量 =========================
def load_glove_embeddings(glove_path, vocab, embed_dim=100):
    """
    加载GloVe预训练词向量, 构建embedding矩阵

    参数:
        glove_path: GloVe文件路径 (如 glove.6B.100d.txt)
        vocab: Vocabulary对象
        embed_dim: GloVe向量维度 (需与文件匹配)

    返回:
        embedding_matrix: (vocab_size, embed_dim) 的 torch.Tensor
        coverage: 词汇表中有GloVe向量的词的比例
    """
    print(f"  加载GloVe词向量: {glove_path}")

    # 1. 读取GloVe文件
    glove_dict = {}
    with open(glove_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            if word in vocab.word2idx:  # 只加载词汇表中的词
                vector = np.array(values[1:], dtype=np.float32)
                if len(vector) == embed_dim:
                    glove_dict[word] = vector

    # 2. 构建embedding矩阵
    embedding_matrix = np.random.uniform(
        -0.25, 0.25, (vocab.vocab_size, embed_dim)
    ).astype(np.float32)
    embedding_matrix[0] = 0  # <PAD> 用零向量

    found = 0
    for word, idx in vocab.word2idx.items():
        if word in glove_dict:
            embedding_matrix[idx] = glove_dict[word]
            found += 1

    coverage = found / vocab.vocab_size
    print(f"  GloVe覆盖率: {found}/{vocab.vocab_size} = {coverage:.2%}")

    return torch.tensor(embedding_matrix), coverage
