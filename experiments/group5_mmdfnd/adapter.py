"""
MMDFND adapter for running the reference implementation.

Provides utilities to:
1. Set up the MMDFND environment
2. Adapt configurations for different datasets
3. Run training and capture metrics
4. (Optional) Adapt MMDFND for English text-only data
"""

import os
import sys
import subprocess
from typing import Dict, Optional


MMDFND_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "MMDFND",
)


MMDFND_REQUIRED_PACKAGES = [
    "torch", "transformers", "pandas", "numpy", "tqdm",
    "sklearn", "positional_encodings", "timm",
]


def check_mmdfnd_requirements():
    """Check which MMDFND dependencies are installed."""
    required = list(MMDFND_REQUIRED_PACKAGES)
    optional = ["cn_clip", "clip"]

    status = {}
    for pkg in required + optional:
        try:
            __import__(pkg.replace("-", "_"))
            status[pkg] = True
        except ImportError:
            status[pkg] = False

    return status


def packages_sufficient_for_training() -> bool:
    """True if core packages needed to run MMDFND main.py are installed."""
    st = check_mmdfnd_requirements()
    return all(st.get(p, False) for p in MMDFND_REQUIRED_PACKAGES)


def check_weibo_pickles_ready() -> bool:
    """Preprocessed tensor pickles expected under MMDFND/data/ (from data_pre.py / clip_data_pre.py)."""
    base = os.path.join(MMDFND_DIR, "data")
    names = [
        "train_loader.pkl", "train_clip_loader.pkl",
        "val_loader.pkl", "val_clip_loader.pkl",
        "test_loader.pkl", "test_clip_loader.pkl",
    ]
    return all(os.path.isfile(os.path.join(base, n)) for n in names)


def print_setup_instructions():
    """Print instructions for setting up the MMDFND environment."""
    print("=" * 60)
    print("  MMDFND Setup Instructions")
    print("=" * 60)
    print()
    print("  1. Install dependencies:")
    print(f"     cd {MMDFND_DIR}")
    print("     pip install -r requirements.txt")
    print()
    print("  2. Download pretrained models:")
    print("     a. Chinese RoBERTa:")
    print("        https://drive.google.com/drive/folders/1y2k22iMG1i1f302NLf-bj7UEe9zwTwLR")
    print("        -> Place in ./pretrained_model/")
    print()
    print("     b. MAE ViT-base:")
    print("        https://github.com/facebookresearch/mae")
    print("        -> Download mae_pretrain_vit_base.pth to MMDFND root")
    print()
    print("     c. Chinese-CLIP ViT-B-16:")
    print("        https://github.com/OFA-Sys/Chinese-CLIP")
    print("        -> Download clip_cn_vit-b-16.pt to MMDFND root")
    print()
    print("  3. Download datasets:")
    print("     a. Weibo dataset:")
    print("        https://pan.baidu.com/s/1TGc-8RUt6BIHO1rjnzuPxQ (code: qwer)")
    print("        -> Place in ./data/")
    print()
    print("     b. Weibo21 dataset:")
    print("        Contact Dr. Qiong Nan (nanqiong19z@ict.ac.cn)")
    print("        -> Place in ./Weibo_21/")
    print()
    print("  4. Preprocess data:")
    print("     python data_pre.py")
    print("     python clip_data_pre.py")
    print("     python weibo21_data_pre.py")
    print("     python weibo21_clip_data_pre.py")
    print()
    print("  5. Train:")
    print("     python main.py --dataset weibo")
    print("     python main.py --dataset weibo21")
    print("=" * 60)


def check_data_ready() -> Dict[str, bool]:
    """Check if MMDFND data files are present."""
    checks = {
        "weibo_train_csv": os.path.exists(os.path.join(MMDFND_DIR, "data", "train_origin.csv")),
        "weibo_train_pkl": os.path.exists(os.path.join(MMDFND_DIR, "data", "train_loader.pkl")),
        "weibo_clip_pkl": os.path.exists(os.path.join(MMDFND_DIR, "data", "train_clip_loader.pkl")),
        "weibo21_xlsx": os.path.exists(os.path.join(MMDFND_DIR, "Weibo_21", "train_datasets.xlsx")),
        "weibo21_pkl": os.path.exists(os.path.join(MMDFND_DIR, "Weibo_21", "train_loader.pkl")),
        "roberta_model": os.path.exists(os.path.join(MMDFND_DIR, "pretrained_model")),
        "mae_weights": os.path.exists(os.path.join(MMDFND_DIR, "mae_pretrain_vit_base.pth")),
        "clip_weights": os.path.exists(os.path.join(MMDFND_DIR, "clip_cn_vit-b-16.pt")),
    }
    return checks


def run_mmdfnd_training(dataset: str = "weibo21", extra_args: Optional[Dict] = None):
    """
    Launch MMDFND training as a subprocess.

    Args:
        dataset: 'weibo' or 'weibo21'
        extra_args: Additional CLI arguments for main.py (e.g. root_path for weibo CSVs)

    Environment:
        MMDFND_DATA_ROOT: If set and extra_args does not include root_path, passed as --root_path
                          (folder containing train_origin.csv / val_origin.csv / test_origin.csv).
    """
    merged = dict(extra_args or {})
    env_root = os.environ.get("MMDFND_DATA_ROOT", "").strip()
    if env_root and "root_path" not in merged:
        merged["root_path"] = env_root

    cmd = [sys.executable, "main.py", "--dataset", dataset]
    for k, v in merged.items():
        cmd.extend([f"--{k}", str(v)])

    print(f"  Running: {' '.join(cmd)}")
    print(f"  Working dir: {MMDFND_DIR}")

    result = subprocess.run(
        cmd, cwd=MMDFND_DIR,
        capture_output=False, text=True,
    )

    return result.returncode


def create_english_adapter_config():
    """
    Generate configuration for adapting MMDFND to English text-only data.

    This is an exploratory adaptation that would require modifying
    the MMDFND source code.
    """
    return {
        "description": "Adapt MMDFND for English text-only fake news detection",
        "modifications": {
            "bert_model": "bert-base-uncased or roberta-base (replace Chinese RoBERTa)",
            "clip_model": "openai/clip-vit-base-patch32 (replace Chinese-CLIP)",
            "image_handling": "Either scrape article images or use text-only mode",
            "domain_labels": "Map English news categories to 9 domain IDs",
            "tokenizer": "Use HuggingFace AutoTokenizer instead of BertTokenizer with Chinese vocab",
            "dataloader": "Adapt utils/clip_dataloader.py for English CSV format",
        },
        "files_to_modify": [
            "MMDFND/main.py - Change bert path and model defaults",
            "MMDFND/run.py - Update category_dict for English domains",
            "MMDFND/utils/clip_dataloader.py - Adapt data loading",
            "MMDFND/model/MMDFND.py - Replace cn_clip with openai CLIP",
        ],
        "notes": [
            "The core architecture (PLE experts, pivot transformer, AdaIN) is language-agnostic",
            "Main challenge is replacing Chinese-specific components (cn_clip, Chinese RoBERTa)",
            "Text-only mode would require removing/zeroing image branches",
        ],
    }
