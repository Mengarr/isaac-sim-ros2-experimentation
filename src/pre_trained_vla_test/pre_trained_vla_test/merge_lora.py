"""
merge_lora.py
-------------
Merges a LoRA adapter (saved by lerobot-train --peft.method_type=LORA) into the
base model and saves a complete, self-contained checkpoint ready for inference.

The saved checkpoint uses the fine-tuned config (correct feature/camera names)
with the merged weights, so PI0Policy.from_pretrained(output) just works.

Usage:
python3 merge_lora.py \
    --base lerobot/pi05_libero \
    --adapter /home/ubuntu/Downloads/pi05_finetuned/checkpoints/003000/pretrained_model \
    --output  /home/ubuntu/checkpoints/pi05_merged
"""

import argparse
import shutil
from pathlib import Path

import torch
from peft import PeftModel

from lerobot.policies.pi0 import PI0Policy

_DEFAULT_BASE = "lerobot/pi05_libero"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default=_DEFAULT_BASE, help="Base model ID or local path")
    parser.add_argument("--adapter", type=Path, required=True, help="Path to the PEFT adapter checkpoint")
    parser.add_argument("--output", type=Path, required=True, help="Directory to write the merged checkpoint")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    print(f"Loading base model from {args.base} ...")
    policy = PI0Policy.from_pretrained(args.base)
    policy.to("cpu")

    print(f"Applying LoRA adapter from {args.adapter} ...")
    policy = PeftModel.from_pretrained(policy, str(args.adapter))
    policy = policy.merge_and_unload()
    print("Adapter merged.")

    # Save using lerobot's own serialisation so key formats (e.g. layernorm
    # .dense.weight) are written correctly for from_pretrained to read back.
    print(f"Saving merged weights to {args.output} ...")
    policy.save_pretrained(args.output)

    # Overwrite config and stats files with the fine-tuned versions so the
    # checkpoint carries the correct feature/camera schema and normalisation
    # stats, not the base model's. Excludes PEFT-specific files.
    skip = {"adapter_config.json", "adapter_model.safetensors", "train_config.json", "README.md"}
    print(f"Copying fine-tuned config and stats from {args.adapter} ...")
    for src in sorted(args.adapter.iterdir()):
        if src.name not in skip:
            shutil.copy(src, args.output / src.name)
            print(f"  {src.name}")

    print("\nDone. Load the merged checkpoint with:")
    print(f"  PI0Policy.from_pretrained('{args.output}')")


if __name__ == "__main__":
    main()
