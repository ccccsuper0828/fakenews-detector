"""
Experiment Group 2: Transformer-based classifiers.

Supports BERT, RoBERTa, DistilBERT with configurable
freeze strategies and classifier heads.
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig


class TransformerClassifier(nn.Module):
    """
    Pretrained transformer + classification head for fake news detection.

    Args:
        model_name: HuggingFace model identifier.
        num_labels: Number of output labels (1 for binary with BCE).
        freeze_strategy: 'none' | 'all' | 'top_n'.
        unfreeze_top_n: Number of top transformer layers to keep unfrozen
                        when freeze_strategy='top_n'.
        classifier_hidden: Hidden dim of the classification MLP.
        classifier_dropout: Dropout in the classifier.
    """

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        num_labels: int = 1,
        freeze_strategy: str = "none",
        unfreeze_top_n: int = 4,
        classifier_hidden: int = 256,
        classifier_dropout: float = 0.3,
    ):
        super().__init__()
        self.model_name = model_name
        self.config = AutoConfig.from_pretrained(model_name)
        self.transformer = AutoModel.from_pretrained(model_name)
        hidden_size = self.config.hidden_size

        self._apply_freeze(freeze_strategy, unfreeze_top_n)

        self.classifier = nn.Sequential(
            nn.Dropout(classifier_dropout),
            nn.Linear(hidden_size, classifier_hidden),
            nn.GELU(),
            nn.Dropout(classifier_dropout * 0.5),
            nn.Linear(classifier_hidden, num_labels),
        )

    def _apply_freeze(self, strategy: str, top_n: int):
        if strategy == "all":
            for param in self.transformer.parameters():
                param.requires_grad = False
        elif strategy == "top_n":
            for param in self.transformer.parameters():
                param.requires_grad = False
            # Unfreeze top N encoder layers
            if hasattr(self.transformer, "encoder"):
                layers = self.transformer.encoder.layer
            elif hasattr(self.transformer, "transformer"):
                layers = self.transformer.transformer.layer
            else:
                return
            for layer in layers[-top_n:]:
                for param in layer.parameters():
                    param.requires_grad = True
            # Always unfreeze pooler if present
            if hasattr(self.transformer, "pooler") and self.transformer.pooler is not None:
                for param in self.transformer.pooler.parameters():
                    param.requires_grad = True

    def forward(self, input_ids, attention_mask=None, return_hidden=False):
        """
        Args:
            input_ids: (B, T) token IDs.
            attention_mask: (B, T) mask (1 = real token, 0 = pad).
            return_hidden: If True, also return the [CLS] hidden state.
        Returns:
            logits: (B,) or (B, num_labels).
        """
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        # Use [CLS] token representation
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            pooled = outputs.pooler_output
        else:
            pooled = outputs.last_hidden_state[:, 0, :]

        logits = self.classifier(pooled).squeeze(-1)

        if return_hidden:
            return logits, pooled
        return logits

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class TransformerDataset(torch.utils.data.Dataset):
    """Dataset that tokenizes texts on-the-fly for transformer models."""

    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return (
            encoding["input_ids"].squeeze(0),
            encoding["attention_mask"].squeeze(0),
            torch.tensor(label, dtype=torch.float),
        )
