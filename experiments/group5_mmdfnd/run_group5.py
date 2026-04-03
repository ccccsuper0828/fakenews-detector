"""
Group 5 Runner: MMDFND paper reproduction.

Usage:
    python -m experiments.group5_mmdfnd.run_group5 [--check|--setup|--train|--all]

Weibo CSV 在网盘/Colab（例如含 train_origin.csv 的文件夹 ``data 3``）::

    python -m experiments.group5_mmdfnd.run_group5 --train --dataset weibo \\
        --data-root "/content/drive/MyDrive/data 3"

等价环境变量：``MMDFND_DATA_ROOT=/path/to/folder``（再执行训练子进程时会带上 ``--root_path``）。

注意：训练仍依赖已在 ``MMDFND/data/`` 下生成好的 ``*_loader.pkl``（需先在 ``MMDFND/`` 下运行
``data_pre.py`` / ``clip_data_pre.py``，且图片目录 ``data/nonrumor_images``、``data/rumor_images``
需齐全；预处理脚本默认读 ``MMDFND/data/train_origin.csv``，可把该目录指到网盘或先复制 CSV）。
"""

import os
import sys
import argparse
import json
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from experiments.group5_mmdfnd.adapter import (
    check_mmdfnd_requirements,
    check_data_ready,
    print_setup_instructions,
    run_mmdfnd_training,
    create_english_adapter_config,
    MMDFND_DIR,
    packages_sufficient_for_training,
    check_weibo_pickles_ready,
)


def cmd_check():
    """Check environment readiness."""
    print("=" * 60)
    print("  MMDFND Environment Check")
    print("=" * 60)

    print("\n  Python packages:")
    pkg_status = check_mmdfnd_requirements()
    for pkg, installed in pkg_status.items():
        status = "OK" if installed else "MISSING"
        print(f"    {pkg:25s} [{status}]")

    print("\n  Data and model files:")
    data_status = check_data_ready()
    for item, ready in data_status.items():
        status = "OK" if ready else "NOT FOUND"
        print(f"    {item:25s} [{status}]")

    all_ready = all(pkg_status.values()) and all(data_status.values())
    print(f"\n  Overall status: {'READY' if all_ready else 'NOT READY'}")

    if not all_ready:
        missing_pkgs = [k for k, v in pkg_status.items() if not v]
        missing_data = [k for k, v in data_status.items() if not v]
        if missing_pkgs:
            print(f"\n  Missing packages: {', '.join(missing_pkgs)}")
            print(f"  Run: pip install -r {os.path.join(MMDFND_DIR, 'requirements.txt')}")
        if missing_data:
            print(f"\n  Missing data/models: {', '.join(missing_data)}")
            print("  Run: python -m experiments.group5_mmdfnd.run_group5 --setup")

    return all_ready


def cmd_setup():
    """Print setup instructions."""
    print_setup_instructions()


def cmd_train(dataset: str = "weibo21", epochs: int = 50, data_root: Optional[str] = None):
    """Run MMDFND training."""
    print("=" * 60)
    print(f"  MMDFND Training - Dataset: {dataset}")
    print("=" * 60)

    if not data_root:
        env_root = (os.environ.get("MMDFND_DATA_ROOT") or "").strip()
        if env_root:
            data_root = env_root

    extra = {"epoch": epochs}
    if data_root:
        if dataset != "weibo":
            print("\n  --data-root only applies to --dataset weibo (CSV: train/val/test_origin.csv).")
            return
        if not packages_sufficient_for_training():
            print("\n  Missing Python packages. Run: pip install -r MMDFND/requirements.txt")
            return
        if not check_weibo_pickles_ready():
            print("\n  Missing MMDFND/data/*_loader.pkl — run preprocessing first from MMDFND/:")
            print("    python data_pre.py && python clip_data_pre.py")
            print("  (Image folders data/nonrumor_images/ and data/rumor_images/ must exist under MMDFND.)")
            return
        extra["root_path"] = data_root
        print(f"  Weibo CSV root: {data_root}")
    elif not cmd_check():
        print("\n  Environment not ready. Please run --setup first.")
        return

    print(f"\n  Starting training with dataset={dataset}, epochs={epochs}")
    ret = run_mmdfnd_training(
        dataset=dataset,
        extra_args=extra,
    )

    if ret == 0:
        print("\n  Training completed successfully!")
        print("  Check MMDFND/parameter_mmdfnd.pkl for saved model")
    else:
        print(f"\n  Training failed with return code {ret}")


def cmd_adapter():
    """Show English adaptation plan."""
    print("=" * 60)
    print("  MMDFND English Adaptation Plan")
    print("=" * 60)

    config = create_english_adapter_config()
    print(f"\n  {config['description']}\n")

    print("  Required modifications:")
    for key, desc in config["modifications"].items():
        print(f"    - {key}: {desc}")

    print("\n  Files to modify:")
    for f in config["files_to_modify"]:
        print(f"    - {f}")

    print("\n  Notes:")
    for note in config["notes"]:
        print(f"    - {note}")

    output_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "outputs", "group5_mmdfnd",
    )
    os.makedirs(output_dir, exist_ok=True)
    config_path = os.path.join(output_dir, "english_adapter_plan.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"\n  Plan saved to {config_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="Check environment")
    parser.add_argument("--setup", action="store_true", help="Show setup instructions")
    parser.add_argument("--train", action="store_true", help="Run training")
    parser.add_argument("--adapter", action="store_true", help="Show English adapter plan")
    parser.add_argument("--dataset", default="weibo21", choices=["weibo", "weibo21"])
    parser.add_argument(
        "--data-root",
        default=None,
        metavar="DIR",
        help="Weibo only: directory with train_origin.csv, val_origin.csv, test_origin.csv "
        "(e.g. Colab: /content/drive/MyDrive/data 3). Forwards to MMDFND main.py --root_path. "
        "Overrides MMDFND_DATA_ROOT if both are set.",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--all", action="store_true", help="Run check + setup")
    args = parser.parse_args()

    if args.all or (not any([args.check, args.setup, args.train, args.adapter])):
        cmd_check()
        print()
        cmd_setup()
        return

    if args.check:
        cmd_check()
    if args.setup:
        cmd_setup()
    if args.train:
        cmd_train(args.dataset, args.epochs, data_root=args.data_root)
    if args.adapter:
        cmd_adapter()


if __name__ == "__main__":
    main()
