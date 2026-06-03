"""
merge_lora.py
-------------
Merges a LoRA adapter (saved by lerobot-train --peft.method_type=LORA) into the
base model and saves a complete, self-contained checkpoint ready for inference.

The saved checkpoint uses the fine-tuned config (correct feature/camera names)
with the merged weights, so PI0Policy.from_pretrained(output) just works.

Usage:
    python merge_lora.py \
        --base lerobot/pi05_libero \
        --adapter /home/ubuntu/Downloads/pi05_finetuned/checkpoints/003000/pretrained_model \
        --output  /home/ubuntu/checkpoints/pi05_merged

    # Optionally override the base model (default: lerobot/pi05_libero):
    python merge_lora.py --base lerobot/pi05_libero --adapter ... --output ...
"""

import argparse
import shutil
from pathlib import Path

import torch
from peft import PeftModel
from safetensors.torch import save_file

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
    policy.model = PeftModel.from_pretrained(policy.model, str(args.adapter))
    policy.model = policy.model.merge_and_unload()
    print("Adapter merged.")

    # Copy config files from the fine-tuned adapter checkpoint so the saved
    # checkpoint carries the correct feature/camera schema, not the base model's.
    config_files = [
        "config.json",
        "policy_preprocessor.json",
        "policy_postprocessor.json",
    ]
    print(f"Copying fine-tuned config from {args.adapter} ...")
    for filename in config_files:
        src = args.adapter / filename
        if src.exists():
            shutil.copy(src, args.output / filename)
            print(f"  {filename}")
        else:
            print(f"  {filename} not found in adapter path — skipping")

    weights_path = args.output / "model.safetensors"
    print(f"Saving merged weights to {weights_path} ...")
    state_dict = {k: v.contiguous() for k, v in policy.state_dict().items()}
    save_file(state_dict, str(weights_path))

    print("\nDone. Load the merged checkpoint with:")
    print(f"  PI0Policy.from_pretrained('{args.output}')")


if __name__ == "__main__":
    main()
