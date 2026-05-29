"""
quantize_pi0.py
---------------
Loads lerobot/pi0_base (FP32, ~13 GiB RAM) on CPU, converts weights to FP16,
and saves a new safetensors checkpoint (~6.5 GiB).

Run this on any machine with >14 GiB RAM — no GPU required.

Usage:
    python quantize_pi0.py
    python quantize_pi0.py --output ~/checkpoints/pi0_base_fp16.safetensors
"""

import argparse
from pathlib import Path

import torch
from safetensors.torch import save_file

from lerobot.policies.pi0 import PI0Policy

_MODEL_ID = "lerobot/pi0_base"
_DEFAULT_OUTPUT = Path.home() / "checkpoints" / "pi0_base_fp16.safetensors"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=_DEFAULT_OUTPUT,
        help="Path to save the FP16 safetensors file",
    )
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading {_MODEL_ID} on CPU (FP32, ~13 GiB RAM)...")
    policy = PI0Policy.from_pretrained(_MODEL_ID)

    total_fp32 = sum(p.numel() * p.element_size() for p in policy.parameters()) / 1024**3
    print(f"Loaded. Weight footprint: {total_fp32:.2f} GiB (FP32)")

    print("Converting to FP16...")
    policy.half()

    total_fp16 = sum(p.numel() * p.element_size() for p in policy.parameters()) / 1024**3
    print(f"Converted. Weight footprint: {total_fp16:.2f} GiB (FP16)")

    print(f"Saving to {args.output} ...")
    state_dict = {k: v for k, v in policy.state_dict().items()}
    save_file(state_dict, str(args.output))

    print("Done.")
    print(f"Load the quantized checkpoint with:")
    print(f"  PI0Policy.from_pretrained('{args.output}')")


if __name__ == "__main__":
    main()
