# 🔍 Fake News Detector

**CDS525 Group Project** — Deep Learning-based Fake News Detection

Two model architectures with comprehensive experiments, visualizations, and explainability.

| Model | Test Accuracy | Parameters | Training Time (T4 GPU) |
|-------|:------------:|:----------:|:---------------------:|
| **BiLSTM + Attention + GloVe** | **97.35%** | ~0.5M trainable | ~3 min/experiment |
| **DistilBERT (Transformer)** | **~99%** | 66M (fine-tuned) | ~8 min/experiment |

---

## 📁 Project Structure

```
fakenews-detector/
├── src/                          # Core modules (BiLSTM project)
│   ├── data_utils.py             # Data loading, cleaning, vocabulary, GloVe
│   ├── data_augment.py           # External dataset loading & EDA augmentation
│   ├── model.py                  # BiLSTM + Attention classifier
│   ├── trainer.py                # Training loop, FocalLoss, early stopping
│   ├── visualize.py              # All 8 required figures
│   └── chain_of_thought.py       # Attention-based explainability (CoT)
├── main.py                       # Local entry point (runs all experiments)
├── FakeNewsDetection_Colab.ipynb  # ★ BiLSTM — Google Colab notebook
├── BERT_FakeNews_Colab.ipynb      # ★ DistilBERT — Google Colab notebook
├── requirements.txt              # Python dependencies
└── README.md
```

---

## 📦 Dataset

The project uses **3 datasets** merged together (~50K total samples):

### 1. Main Dataset — `fakenews 2.csv` (4,987 rows)

- **Format**: `text, label` (0=Fake, 1=Real)
- **Source**: Course-provided dataset
- **Download**: Available from the course materials, or use a similar fake news dataset from [Kaggle](https://www.kaggle.com/)

### 2. Extra Dataset — `News _dataset/` (44,898 rows)

This is the **Kaggle Fake and Real News Dataset** by Clément Bisaillon.

| File | Rows | Label |
|------|:----:|:-----:|
| `Fake.csv` | 23,481 | 0 (Fake) |
| `True.csv` | 21,417 | 1 (Real) |

- **Format**: `title, text, subject, date`
- **Download from Kaggle**: [https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)

#### Download Steps:
1. Go to the Kaggle link above → Click **Download** (需要 Kaggle 账号)
2. Unzip to get `Fake.csv` and `True.csv`
3. Place them in a folder called `News _dataset/` (note the space before underscore)

### 3. GloVe Pre-trained Embeddings (BiLSTM only)

- **File**: `glove.6B.100d.txt` (347 MB)
- **Download**: [https://nlp.stanford.edu/data/glove.6B.zip](https://nlp.stanford.edu/data/glove.6B.zip) (822 MB zip)
- Extract `glove.6B.100d.txt` from the zip file
- The Colab notebook **downloads this automatically** — no manual action needed

### Final Data Directory Layout

```
project_root/
├── fakenews 2.csv
├── News _dataset/
│   ├── Fake.csv
│   └── True.csv
└── glove.6B.100d.txt          # BiLSTM only; auto-downloaded on Colab
```

After merging & deduplication: **43,458 unique samples** (20,733 Fake + 22,720 Real)

With data augmentation (EDA): **60,832 training samples**

---

## 🚀 How to Run

### Option A: Google Colab (Recommended)

> **No local setup needed.** Just upload data to Google Drive and run.

#### BiLSTM + GloVe:
1. Upload `fakenews 2.csv` and `News _dataset/` folder to **Google Drive** → `My Drive/fakenews/`
2. Open `FakeNewsDetection_Colab.ipynb` in Colab
3. Set Runtime → **GPU (T4)**
4. Run all cells (GloVe is auto-downloaded)
5. ~30 min total for all experiments

#### DistilBERT:
1. Same data setup as above
2. Open `BERT_FakeNews_Colab.ipynb` in Colab
3. Set Runtime → **GPU (T4)** (A100 recommended for faster training)
4. Run all cells
5. ~60 min total for all experiments

### Option B: Local Machine

```bash
# 1. Clone the repository
git clone https://github.com/ccccsuper0828/fakenews-detector.git
cd fakenews-detector

# 2. Install dependencies
pip install -r requirements.txt

# 3. Place data files (see Dataset section above)
#    - fakenews 2.csv
#    - News _dataset/Fake.csv, News _dataset/True.csv
#    - glove.6B.100d.txt

# 4. Run all experiments
python main.py
```

**Requirements**: Python 3.8+, PyTorch 2.0+, CUDA GPU recommended (CPU works but ~10x slower)

---

## 🏗️ Model Architectures

### Model 1: BiLSTM + Attention + GloVe

```
Input Text → Tokenize → GloVe Embedding (100d, frozen)
  → BiLSTM (2 layers, 128 hidden, bidirectional)
  → Attention Mechanism (2-layer MLP)
  → Weighted Context Vector (256d)
  → FC(256→64) → ReLU → Dropout → FC(64→1) → Sigmoid
```

**Key Features**:
- GloVe pre-trained embeddings (98.5% vocabulary coverage)
- Frozen embeddings → reduces 2M trainable params, prevents overfitting
- Smart truncation: keeps 70% head + 30% tail of long articles
- Stopword removal for cleaner signal

### Model 2: DistilBERT (Transformer)

```
Input Text → BERT Tokenizer (subword, max 256 tokens)
  → DistilBERT (6 layers, 12 attention heads, 768d)
  → [CLS] Token Representation
  → FC(768→256) → ReLU → Dropout → FC(256→1) → Sigmoid
```

**Key Features**:
- Pre-trained on BookCorpus + Wikipedia (3.3B words)
- Minimal text cleaning (BERT needs punctuation & stopwords for context)
- "Reuters" text removed to prevent data leakage
- Linear warmup scheduler (standard for Transformer fine-tuning)

---

## ⚙️ Training Strategy

| Component | BiLSTM | DistilBERT |
|-----------|--------|------------|
| **Optimizer** | AdamW (lr=0.001, wd=1e-4) | AdamW (lr=2e-5, wd=0.01) |
| **LR Scheduler** | ReduceLROnPlateau | Linear Warmup (10%) |
| **Loss Functions** | BCE / Focal Loss | BCE / Focal Loss |
| **Regularization** | Dropout(0.5), Frozen GloVe | Dropout(0.2) |
| **Early Stopping** | Patience=5 on val_acc | Patience=3 on val_acc |
| **Gradient Clipping** | max_norm=1.0 | max_norm=1.0 |
| **Epochs** | 20 (max) | 4 (max) |
| **Data Augmentation** | EDA (random delete/swap) | EDA (random delete/swap) |

---

## 📊 Experiments & Figures

Both notebooks generate **8 required figures** for the course report:

| Figure | Description |
|--------|-------------|
| **Fig 1** | Training curves — BCE Loss (default config) |
| **Fig 2** | Training curves — Focal Loss (default config) |
| **Fig 3** | Learning Rate comparison — BCE Loss (4 LRs) |
| **Fig 4** | Learning Rate comparison — Focal Loss (4 LRs) |
| **Fig 5** | Batch Size comparison — BCE Loss (4 sizes) |
| **Fig 6** | Batch Size comparison — Focal Loss (4 sizes) |
| **Fig 7** | Test Predictions table (top 100 samples) |
| **Fig 8** | Confusion Matrix |
| **Fig 9** | ROC-AUC Curve (DistilBERT only) |

---

## 🧠 Chain-of-Thought (CoT) Explainability

Both models include a **Chain-of-Thought reasoning module** that explains predictions:

```
Step 1 - Text Feature Analysis:
  [Sensational Language] LOW
  [Source Credibility]   HIGH
    Found: according to, officials said
  [Emotional Tone]      LOW
  [Clickbait Pattern]   LOW

Step 2 - Model Attention Key Words:
  'reuters'     [0.0812] ████████████████
  'officials'   [0.0654] █████████████
  'government'  [0.0531] ██████████

Step 3 - Reasoning Chain:
  1. The text references credible sources, consistent with legitimate news.
  2. The text uses neutral, factual language consistent with professional journalism.
  3. The text maintains an objective tone without excessive emotional manipulation.

Conclusion: This article is classified as [Real] with 99.87% confidence.
```

The CoT module analyzes:
1. **Text features**: sensational words, credibility indicators, emotional tone, clickbait patterns
2. **Model attention weights**: which words/tokens the model focused on most
3. **Reasoning chain**: step-by-step logical explanation combining both signals

---

## 📈 Results Summary (BiLSTM on Colab T4 GPU)

| Config | Val Acc | Test Acc |
|--------|:-------:|:--------:|
| BCE, LR=0.001, BS=32 (default) | 97.03% | 97.23% |
| Focal, LR=0.001, BS=32 | 97.08% | 97.18% |
| BCE, LR=0.001, **BS=32** (best) | **97.22%** | **97.35%** |
| BCE, LR=0.01 (too high) | NaN | Crashed |
| BCE, LR=0.0001 | ~96.5% | ~96.5% |

---

## 🛠️ Dependencies

```
torch>=2.0.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
nltk>=3.8.0
tqdm>=4.65.0
transformers          # DistilBERT notebook only
accelerate            # DistilBERT notebook only
```

---

## 📚 References

- **GloVe**: Pennington et al., "GloVe: Global Vectors for Word Representation", EMNLP 2014
- **EDA**: Jason Wei & Kai Zou, "EDA: Easy Data Augmentation Techniques for Boosting Performance on Text Classification Tasks", 2019
- **DistilBERT**: Sanh et al., "DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter", NeurIPS 2019 Workshop
- **Focal Loss**: Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017
- **Dataset**: Clément Bisaillon, "Fake and Real News Dataset", Kaggle

---

## 📄 License

This project is for academic purposes (CDS525 Course Project).
