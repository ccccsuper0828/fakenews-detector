"""
模型定义模块
============
BiLSTM + Attention 分类器:
  - Embedding层: 支持GloVe预训练词向量
  - 双向LSTM: 捕获上下文语义
  - Attention: 聚焦关键词
  - 全连接层 + Dropout: 分类
"""

import torch
import torch.nn as nn
import numpy as np


class BiLSTMAttentionClassifier(nn.Module):
    """
    双向LSTM + Attention 假新闻分类器

    ★ 核心改进: 支持GloVe预训练词向量
      - 随机初始化的embedding需要从5000条数据学语义 → 很难
      - GloVe embedding从60亿词中学到的语义 → 直接用, 大幅提升效果
    """

    def __init__(self, vocab_size, embed_dim=100, hidden_dim=128,
                 num_layers=2, dropout=0.5, pad_idx=0,
                 pretrained_embeddings=None, freeze_embeddings=False):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)

        # ★ 加载预训练词向量
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)
            if freeze_embeddings:
                self.embedding.weight.requires_grad = False  # 冻结: 不微调GloVe

        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )

        lstm_out_dim = hidden_dim * 2  # 双向

        # Attention (两层MLP)
        self.attention = nn.Sequential(
            nn.Linear(lstm_out_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

        # 分类头
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_out_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

        self.dropout_emb = nn.Dropout(0.2)

    def forward(self, x, return_attention=False):
        """
        前向传播

        参数:
            x: (batch_size, seq_len) 词索引张量
            return_attention: 是否同时返回attention权重

        返回:
            logits: (batch_size,) 未经sigmoid的原始分数
            attention_weights: (batch_size, seq_len) [可选]
        """
        # 1. Embedding
        embedded = self.dropout_emb(self.embedding(x))  # (batch, seq, embed)

        # 2. BiLSTM
        lstm_out, _ = self.lstm(embedded)  # (batch, seq, hidden*2)

        # 3. Attention
        attention_scores = self.attention(lstm_out).squeeze(-1)  # (batch, seq)

        # Mask padding位置
        mask = (x != 0).float()
        attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        attention_weights = torch.softmax(attention_scores, dim=1)

        # 加权求和
        context = torch.bmm(
            attention_weights.unsqueeze(1), lstm_out
        ).squeeze(1)  # (batch, hidden*2)

        # 4. 分类
        logits = self.fc(context).squeeze(-1)  # (batch,)

        if return_attention:
            return logits, attention_weights
        return logits

    def count_parameters(self):
        """统计可训练参数总数"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
