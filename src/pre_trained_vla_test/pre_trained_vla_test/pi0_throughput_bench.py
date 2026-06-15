"""
pi0_throughput_bench.py
-----------------------
Standalone inference throughput benchmark for PI0Policy / PI05Policy.
No ROS, no real sensors — feeds dummy tensors and prints instantaneous
inference frequency after each forward pass.

Usage:
    python pi0_throughput_bench.py [--model_type pi05] [--model_path lerobot/pi05_libero]
                                   [--lora_adapter_path /path/to/adapter]
                                   [--iterations 0]   # 0 = run forever
"""

import argparse
import time

import torch

from lerobot.policies import make_pre_post_processors
from lerobot.policies.pi0 import PI0Policy
from lerobot.policies.pi05 import PI05Policy

_POLICY_CLASSES = {
    "pi0": PI0Policy,
    "pi05": PI05Policy,
}

_DEFAULT_MODEL_PATHS = {
    "pi0": "lerobot/pi0_base",
    "pi05": "lerobot/pi05_libero",
}

_NUM_JOINTS = 6
_IMG_H, _IMG_W = 224, 224
_PROMPT = "pick up the object"


def build_dummy_batch(device: torch.device) -> dict:
    """Return a raw observation dict with dummy tensors (same shapes as the real node)."""
    return {
        "observation.state": torch.zeros(1, _NUM_JOINTS, dtype=torch.float32),
        "observation.images.wrist": torch.randint(0, 256, (1, 3, _IMG_H, _IMG_W), dtype=torch.uint8),
        "observation.images.base": torch.randint(0, 256, (1, 3, _IMG_H, _IMG_W), dtype=torch.uint8),
        "task": _PROMPT,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", default="pi05", choices=list(_POLICY_CLASSES))
    parser.add_argument("--model_path", default="")
    parser.add_argument("--lora_adapter_path", default="")
    parser.add_argument("--iterations", type=int, default=0, help="0 = run forever")
    args = parser.parse_args()

    model_path = args.model_path or _DEFAULT_MODEL_PATHS[args.model_type]
    PolicyClass = _POLICY_CLASSES[args.model_type]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}")

    if args.lora_adapter_path:
        from peft import PeftConfig, PeftModel
        peft_config = PeftConfig.from_pretrained(args.lora_adapter_path)
        base_path = peft_config.base_model_name_or_path
        print(f"Loading base model from {base_path} ...")
        policy = PolicyClass.from_pretrained(base_path)
        print(f"Applying LoRA adapter from {args.lora_adapter_path} ...")
        policy = PeftModel.from_pretrained(
            policy, args.lora_adapter_path, config=peft_config, is_trainable=False
        )
    else:
        print(f"Loading {args.model_type} from {model_path} ...")
        policy = PolicyClass.from_pretrained(model_path)

    policy.eval()
    policy.to(device)

    stats_path = args.lora_adapter_path if args.lora_adapter_path else model_path
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config, pretrained_path=stats_path
    )

    print("Model ready. Starting benchmark (Ctrl-C to stop).\n")

    iteration = 0
    t_prev = None

    while True:
        policy.reset()

        raw_obs = build_dummy_batch(device)
        batch = preprocessor(raw_obs)
        batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

        t0 = time.perf_counter()
        with torch.no_grad():
            actions_raw = policy.predict_action_chunk(batch)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        inference_ms = (t1 - t0) * 1000.0

        if t_prev is not None:
            loop_hz = 1.0 / (t1 - t_prev)
            print(f"[iter {iteration:4d}]  inference: {inference_ms:7.1f} ms   loop: {loop_hz:.2f} Hz")
        else:
            print(f"[iter {iteration:4d}]  inference: {inference_ms:7.1f} ms   loop: -- (first)")

        t_prev = t1
        iteration += 1

        if args.iterations and iteration >= args.iterations:
            break

    print("\nDone.")


if __name__ == "__main__":
    main()
