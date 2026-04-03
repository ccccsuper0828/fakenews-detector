"""
Unified experiment configuration.

Provides dataclass-based configs for all experiment groups,
with YAML serialization support.
"""

import os
import json
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any


@dataclass
class DataConfig:
    csv_path: str = "fakenews 2.csv"
    extra_dataset_dir: Optional[str] = "News _dataset"
    max_vocab_size: int = 20000
    max_length: int = 500
    test_size: float = 0.2
    val_size: float = 0.1
    random_state: int = 42
    augment: bool = True
    num_aug: int = 1


@dataclass
class TrainConfig:
    num_epochs: int = 20
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    early_stop_patience: int = 5
    use_scheduler: bool = True
    scheduler_factor: float = 0.5
    scheduler_patience: int = 3
    grad_clip_norm: float = 1.0
    loss_fn: str = "bce"  # bce | focal
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    device: str = "auto"  # auto | cpu | cuda | mps

    def resolve_device(self) -> str:
        import torch
        if self.device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
            return "cpu"
        return self.device


@dataclass
class RNNConfig:
    rnn_type: str = "bilstm"  # lstm | bilstm | gru | bigru
    embed_dim: int = 100
    hidden_dim: int = 128
    num_layers: int = 2
    dropout: float = 0.5
    attention_type: str = "additive"  # none | additive | dot | multihead
    pooling: str = "attention"  # last | mean | max | attention
    num_attention_heads: int = 4
    freeze_embeddings: bool = True
    embedding_type: str = "glove100"  # random | glove100 | glove300


@dataclass
class TransformerConfig:
    model_name: str = "bert-base-uncased"
    max_seq_length: int = 256
    freeze_strategy: str = "none"  # none | all | top_n | lora
    unfreeze_top_n: int = 4
    classifier_hidden: int = 256
    classifier_dropout: float = 0.3
    warmup_ratio: float = 0.1


@dataclass
class MultiAgentConfig:
    llm_provider: str = "openai"  # openai | local
    llm_model: str = "gpt-4o-mini"
    search_provider: str = "tavily"  # tavily | serpapi | google
    search_top_k: int = 5
    nli_model: str = "roberta-large-mnli"
    judge_mode: str = "llm"  # rule | llm
    api_key_env: str = "OPENAI_API_KEY"
    search_api_key_env: str = "TAVILY_API_KEY"


@dataclass
class KGRLConfig:
    kg_source: str = "conceptnet"  # conceptnet | wikidata | dbpedia
    triple_extractor: str = "llm"  # llm | openie
    rl_algorithm: str = "ppo"  # ppo | dqn | meta_classifier
    rl_hidden_dim: int = 128
    rl_lr: float = 3e-4
    rl_gamma: float = 0.99
    step_cost: float = 0.1
    max_steps: int = 5
    meta_classifier_type: str = "mlp"  # lr | mlp


@dataclass
class ExperimentConfig:
    name: str = "default"
    group: str = "group1_rnn"
    output_dir: str = "outputs"
    seed: int = 42

    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    rnn: RNNConfig = field(default_factory=RNNConfig)
    transformer: TransformerConfig = field(default_factory=TransformerConfig)
    multiagent: MultiAgentConfig = field(default_factory=MultiAgentConfig)
    kg_rl: KGRLConfig = field(default_factory=KGRLConfig)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ExperimentConfig":
        data = DataConfig(**d.pop("data", {}))
        train = TrainConfig(**d.pop("train", {}))
        rnn = RNNConfig(**d.pop("rnn", {}))
        transformer = TransformerConfig(**d.pop("transformer", {}))
        multiagent = MultiAgentConfig(**d.pop("multiagent", {}))
        kg_rl = KGRLConfig(**d.pop("kg_rl", {}))
        return cls(data=data, train=train, rnn=rnn, transformer=transformer,
                   multiagent=multiagent, kg_rl=kg_rl, **d)

    @classmethod
    def load(cls, path: str) -> "ExperimentConfig":
        with open(path) as f:
            return cls.from_dict(json.load(f))
