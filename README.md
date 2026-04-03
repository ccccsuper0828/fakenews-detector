# Fake News Detection — Comparative Study

**CDS525 Group Project** — Five approaches to fake news detection, from RNN to multimodal.

---

## Results Overview

| Group | Method | Best F1 | Accuracy | AUC |
|:-----:|--------|:-------:|:--------:|:---:|
| G0 | BiLSTM Baseline | 0.9732 | 0.9723 | — |
| G1 | RNN Ablation (29 experiments) | 0.9745 | 0.9737 | 0.9969 |
| G2 | LLM Fine-tuning (10 experiments) | **0.9781** | **0.9773** | **0.9982** |
| G3 | Multi-Agent FactAgent (9 iterations) | 0.5410 | 0.4400 | 0.4569 |
| G4 | KG + RL Fusion (9 experiments) | 0.5714 | 0.7000 | 0.7464 |
| G5 | MMDFND Multimodal (Weibo) | 0.9003 | 0.9003 | 0.9595 |

---

## Project Structure

```
fakenews-detector/
├── src/                                  # Core modules
│   ├── data_utils.py                     # Data loading, cleaning, vocabulary, GloVe
│   ├── data_augment.py                   # External dataset merge & EDA augmentation
│   ├── model.py                          # BiLSTM + Attention classifier
│   ├── trainer.py                        # Training loop, FocalLoss, early stopping
│   ├── visualize.py                      # Figure generation
│   └── chain_of_thought.py              # Attention-based explainability
│
├── experiments/                          # Five experiment groups
│   ├── config.py                         # Unified configuration (DataConfig, TrainConfig, etc.)
│   ├── metrics.py                        # Unified evaluation metrics
│   ├── runner.py                         # Shared training runner
│   ├── evaluate_all.py                   # Cross-group comparison
│   │
│   ├── group1_rnn/                       # G1: RNN architecture ablation
│   │   ├── models.py                     # LSTM/BiLSTM/GRU/BiGRU + 4 attention types
│   │   └── run_group1.py                 # 29 ablation experiments
│   │
│   ├── group2_llm/                       # G2: LLM fine-tuning
│   │   ├── models.py                     # BERT/DistilBERT/RoBERTa + freeze strategies
│   │   └── run_group2.py                 # 10 fine-tuning experiments
│   │
│   ├── group3_multiagent/                # G3: FactAgent multi-tool verification
│   │   ├── agents.py                     # 6 tools: Phrase/Language/Commonsense/URL/Standing/Search
│   │   ├── orchestrator.py               # Pipeline: Claim→Search→NLI→Judge
│   │   ├── search_tools.py              # Tavily/SerpAPI/DuckDuckGo/Mock
│   │   └── run_group3.py                # 9 iterative improvements + GridSearch
│   │
│   ├── group4_kg_rl/                     # G4: Knowledge Graph + Reinforcement Learning
│   │   ├── kg_builder.py                 # Triple extraction + ConceptNet/Wikidata
│   │   ├── commonsense_checker.py       # KG feature extraction
│   │   ├── rl_agent.py                   # DQN agent + Meta classifier
│   │   └── run_group4.py                # Meta/DQN/Fusion experiments
│   │
│   └── group5_mmdfnd/                    # G5: MMDFND reproduction + innovations
│       ├── adapter.py                    # MMDFND environment setup
│       ├── innovations.py               # 6 innovation modules (724 lines)
│       └── run_group5.py                # Training runner with data-root support
│
├── MMDFND/                               # MMDFND source code (ACM MM 2024)
│   ├── model/MMDFND.py                   # Multi-Domain PLE + Pivot Transformer
│   ├── run.py                            # Training entry (supports --root_path)
│   ├── main.py                           # CLI entry point
│   └── utils/                            # Data loaders, metrics
│
├── Experiments_Colab.ipynb               # All 5 groups — Colab notebook
├── Experiments_3_5_Colab.ipynb           # Groups 3 & 5 only
├── Experiment5_Colab.ipynb               # Group 5 MMDFND baseline
├── Experiment5_Innovations_Colab.ipynb   # Group 5 innovation ablation
│
├── requirements.txt
└── README.md
```

---

## Five Experiment Groups

### Group 0: Baseline (BiLSTM)
Single BiLSTM + BCE + GloVe, with learning rate and batch size ablation.
- **Result**: Acc=97.23%, F1=97.32%

### Group 1: RNN Architecture Ablation (29 experiments)
Systematic ablation of 4 RNN types × 4 attention mechanisms × 4 pooling strategies, plus hyperparameter and embedding ablation.
- **Best**: BiGRU + Additive Attention → F1=0.9745
- **Finding**: Bidirectional > unidirectional; GRU ≈ LSTM with fewer params

### Group 2: LLM Fine-tuning (10 experiments)
DistilBERT / BERT / RoBERTa × 3 freeze strategies (full / frozen / top-4).
- **Best**: DistilBERT full finetune → F1=0.9781 (3 epochs, smallest model wins)
- **Finding**: top_n4 with lr=0.001 collapses; full finetune with lr=2e-5 is optimal

### Group 3: Multi-Agent Verification — FactAgent (9 iterations)
Zero-training approach inspired by [FactAgent (Li et al., 2024)](https://arxiv.org/abs/2405.01593). Six analysis tools + GridSearch threshold optimization.
- **Tools**: PhraseAnalyzer, LanguageAnalyzer, CommonsenseChecker, URLCredibilityChecker, StandingAnalyzer, SearchNLI
- **Progress**: F1 improved from 0.22 → 0.54 (+145%) over 9 iterations
- **Finding**: Multi-perspective analysis significantly outperforms single NLI

### Group 4: Knowledge Graph + RL (9 experiments)
ConceptNet-based commonsense verification + Meta classifier / DQN routing.
- **Key improvement**: Replaced random noise features with TF-IDF proxy model → AUC from 0.63 → 0.75
- **Finding**: DQN unstable on small datasets; Meta classifier + real features is practical

### Group 5: MMDFND Reproduction + 6 Innovations
Reproduced [MMDFND (Tong et al., ACM MM 2024)](https://dl.acm.org/doi/abs/10.1145/3664647.3681317) on Weibo dataset.
- **Baseline**: Acc=0.90, F1=0.90, AUC=0.96 (matches paper)
- **6 innovations** (code in `innovations.py`): F-EDL uncertainty, cross-modal consistency, auto loss weighting, frequency forensics, expert load balancing, domain-aware contrastive learning

---

## Datasets

### English Dataset (Groups 0-4)
| Source | Samples | Label |
|--------|:-------:|-------|
| Course CSV (`fakenews 2.csv`) | 4,986 | text + label |
| [Kaggle Fake News](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset) | 44,898 | Fake.csv + True.csv |
| **After merge & dedup** | **43,458** | Train 30,416 / Val 4,346 / Test 8,691 |

### Chinese Weibo Dataset (Group 5)
| Split | Samples | Domains | Modality |
|-------|:-------:|:-------:|----------|
| Train | 5,415 | 9 | Text + Image |
| Val | 843 | 9 | Text + Image |
| Test | 1,465 | 9 | Text + Image |

---

## Quick Start

### Local (Group 3 — no GPU needed)
```bash
pip install -r requirements.txt
pip install duckduckgo_search

# Run FactAgent with GridSearch (100 samples, ~20 min)
python -m experiments.group3_multiagent.run_group3 --num-samples 100 --tune
```

### Colab (Groups 1-5)
Upload notebooks to Colab, set Runtime → GPU, run sequentially:

| Notebook | Groups | GPU | Time |
|----------|:------:|:---:|:----:|
| `Experiments_Colab.ipynb` | 1-4 | Recommended | ~3h |
| `Experiment5_Colab.ipynb` | 5 baseline | Required | ~30min |
| `Experiment5_Innovations_Colab.ipynb` | 5 innovations | Required | ~3-5h/exp |

---

## Dependencies

```
torch>=2.0.0
transformers>=4.30.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
nltk>=3.8.0
tqdm>=4.65.0
duckduckgo_search        # Group 3
timm                     # Group 5
positional_encodings     # Group 5
cn_clip                  # Group 5
```

---

## License

Academic use only (CDS525 Course Project).
