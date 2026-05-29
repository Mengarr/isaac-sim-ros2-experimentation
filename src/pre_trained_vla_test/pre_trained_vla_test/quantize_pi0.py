"""
quantize_pi0.py
---------------
Loads lerobot/pi0_base (FP32, ~13 GiB RAM) on CPU, converts weights to FP16,
and saves a complete local checkpoint directory (~6.5 GiB) ready for from_pretrained.

Run this on any machine with >14 GiB RAM — no GPU required.

Usage:
    python quantize_pi0.py
    python quantize_pi0.py --output ~/checkpoints/pi0_base_fp16
"""

import argparse
import glob
import shutil
from pathlib import Path

import torch
from safetensors.torch import save_file

from lerobot.configs import PreTrainedConfig
from lerobot.policies.pi0 import PI0Policy

_MODEL_ID = "lerobot/pi0_base"
_DEFAULT_OUTPUT = Path.home() / "checkpoints" / "pi0_base_fp16"
_CONFIG_FILES = ["config.json", "policy_preprocessor.json", "policy_postprocessor.json", "README.md"]


def find_snapshot_dir() -> Path:
    pattern = str(Path.home() / ".cache/huggingface/hub/models--lerobot--pi0_base/snapshots/*/")
    matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError("pi0_base not found in HuggingFace cache — run the inference node first to trigger the download")
    return Path(matches[0])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=_DEFAULT_OUTPUT,
        help="Directory to save the FP16 checkpoint (default: ~/checkpoints/pi0_base_fp16)",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    snapshot_dir = find_snapshot_dir()
    print(f"Found cached snapshot: {snapshot_dir}")

    print("Copying config files...")
    for filename in _CONFIG_FILES:
        src = snapshot_dir / filename
        if src.exists():
            shutil.copy(src, args.output / filename)
            print(f"  {filename}")

    print(f"\nLoading {_MODEL_ID} on CPU (FP32, ~13 GiB RAM)...")
    config = PreTrainedConfig.from_pretrained(_MODEL_ID)
    config.device = "cpu"
    policy = PI0Policy.from_pretrained(_MODEL_ID, config=config)

    total_fp32 = sum(p.numel() * p.element_size() for p in policy.parameters()) / 1024**3
    print(f"Loaded. Weight footprint: {total_fp32:.2f} GiB (FP32)")

    print("Converting to FP16...")
    policy.half()

    total_fp16 = sum(p.numel() * p.element_size() for p in policy.parameters()) / 1024**3
    print(f"Converted. Weight footprint: {total_fp16:.2f} GiB (FP16)")

    weights_path = args.output / "model.safetensors"
    print(f"\nSaving weights to {weights_path} ...")
    save_file({k: v for k, v in policy.state_dict().items()}, str(weights_path))

    print("\nDone.")
    print(f"Load the quantized checkpoint with:")
    print(f"  PI0Policy.from_pretrained('{args.output}')")


if __name__ == "__main__":
    main()
