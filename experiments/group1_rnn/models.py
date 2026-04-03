"""
Experiment Group 1: Configurable RNN classifiers.

Supports LSTM, BiLSTM, GRU, BiGRU with multiple attention
mechanisms and pooling strategies.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class AdditiveAttention(nn.Module):
    """Two-layer MLP attention (Bahdanau-style)."""

    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, hidden_states, mask=None):
        # hidden_states: (B, T, D)
        scores = self.attn(hidden_states).squeeze(-1)  # (B, T)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        weights = torch.softmax(scores, dim=1)
        context = torch.bmm(weights.unsqueeze(1), hidden_states).squeeze(1)
        return context, weights


class DotProductAttention(nn.Module):
    """Scaled dot-product attention with a learnable query."""

    def __init__(self, input_dim):
        super().__init__()
        self.query = nn.Parameter(torch.randn(input_dim))
        self.scale = math.sqrt(input_dim)

    def forward(self, hidden_states, mask=None):
        scores = torch.matmul(hidden_states, self.query) / self.scale  # (B, T)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        weights = torch.softmax(scores, dim=1)
        context = torch.bmm(weights.unsqueeze(1), hidden_states).squeeze(1)
        return context, weights


class MultiHeadSelfAttention(nn.Module):
    """Standard multi-head self-attention followed by mean pooling."""

    def __init__(self, input_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim=input_dim, num_heads=num_heads,
            dropout=dropout, batch_first=True,
        )
        self.pool_query = nn.Parameter(torch.randn(1, 1, input_dim))

    def forward(self, hidden_states, mask=None):
        key_padding_mask = None
        if mask is not None:
            key_padding_mask = (mask == 0)
        query = self.pool_query.expand(hidden_states.size(0), -1, -1)
        out, weights = self.mha(query, hidden_states, hidden_states,
                                key_padding_mask=key_padding_mask)
        return out.squeeze(1), weights.squeeze(1)


class RNNClassifier(nn.Module):
    """
    Configurable RNN classifier supporting multiple architectures,
    attention mechanisms, and pooling strategies.

    Args:
        vocab_size: Size of the vocabulary.
        embed_dim: Embedding dimension.
        hidden_dim: RNN hidden state dimension.
        num_layers: Number of stacked RNN layers.
        rnn_type: One of 'lstm', 'bilstm', 'gru', 'bigru'.
        attention_type: One of 'none', 'additive', 'dot', 'multihead'.
        pooling: One of 'last', 'mean', 'max', 'attention'.
        num_attention_heads: Heads for multihead attention.
        dropout: Dropout rate.
        pretrained_embeddings: Optional pretrained embedding tensor.
        freeze_embeddings: Whether to freeze the embedding layer.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 100,
        hidden_dim: int = 128,
        num_layers: int = 2,
        rnn_type: str = "bilstm",
        attention_type: str = "additive",
        pooling: str = "attention",
        num_attention_heads: int = 4,
        dropout: float = 0.5,
        pretrained_embeddings=None,
        freeze_embeddings: bool = False,
    ):
        super().__init__()
        self.rnn_type = rnn_type.lower()
        self.attention_type = attention_type.lower()
        self.pooling_type = pooling.lower()

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)
            if freeze_embeddings:
                self.embedding.weight.requires_grad = False

        self.embed_dropout = nn.Dropout(0.2)

        bidirectional = self.rnn_type in ("bilstm", "bigru")
        rnn_cls = nn.LSTM if "lstm" in self.rnn_type else nn.GRU
        self.rnn = rnn_cls(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
        )

        out_dim = hidden_dim * 2 if bidirectional else hidden_dim

        # Attention
        self.attention = None
        if self.pooling_type == "attention" or self.attention_type != "none":
            if self.attention_type == "additive":
                self.attention = AdditiveAttention(out_dim, hidden_dim)
            elif self.attention_type == "dot":
                self.attention = DotProductAttention(out_dim)
            elif self.attention_type == "multihead":
                self.attention = MultiHeadSelfAttention(
                    out_dim, num_heads=num_attention_heads, dropout=dropout
                )
            else:
                self.attention = AdditiveAttention(out_dim, hidden_dim)

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(out_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
        )

    def _pool(self, rnn_out, mask=None):
        if self.pooling_type == "attention" and self.attention is not None:
            ctx, weights = self.attention(rnn_out, mask)
            return ctx
        elif self.pooling_type == "mean":
            if mask is not None:
                mask_f = mask.unsqueeze(-1).float()
                return (rnn_out * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1)
            return rnn_out.mean(dim=1)
        elif self.pooling_type == "max":
            if mask is not None:
                rnn_out = rnn_out.masked_fill(mask.unsqueeze(-1) == 0, -1e9)
            return rnn_out.max(dim=1).values
        else:  # last
            return rnn_out[:, -1, :]

    def forward(self, x, return_attention=False):
        """
        Args:
            x: (B, T) token index tensor.
            return_attention: If True, also return attention weights.
        Returns:
            logits: (B,) raw scores.
            [optional] attention_weights: (B, T).
        """
        mask = (x != 0).float()
        embedded = self.embed_dropout(self.embedding(x))
        rnn_out, _ = self.rnn(embedded)

        if self.pooling_type == "attention" and self.attention is not None:
            context, attn_weights = self.attention(rnn_out, mask)
        else:
            context = self._pool(rnn_out, mask)
            attn_weights = None

        logits = self.classifier(context).squeeze(-1)

        if return_attention and attn_weights is not None:
            return logits, attn_weights
        return logits

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
