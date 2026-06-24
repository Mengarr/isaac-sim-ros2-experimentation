#!/usr/bin/env python
"""Train a LeRobot policy with input-feature dropout.

Drop-in replacement for the `lerobot-train` CLI. Every standard training flag
works exactly the same, because we run LeRobot's *real* `train()` unmodified and
only inject dropout by wrapping `update_policy` (which receives the batch *after*
the preprocessor has run, i.e. already normalized). Wrapping rather than copying
`train()` means this script does not drift when LeRobot's training loop changes.

Dropout semantics (per training step; features are already normalized, so a
"zeroed" feature equals that feature's dataset mean -- the least-informative
imputation):

  Images (cameras, keys under `observation.images.*`):
    * Per sample, with probability `--image_dropout_prob`, exactly ONE camera is
      chosen uniformly at random and blacked out (zeroed).
    * At most one camera is dropped per sample -> the policy always sees >=1 image.

  Proprioception (`observation.state`):
    * Per sample, dropped with probability `--state_dropout_prob`.
    * `--state_dropout_mode=vector`      -> zero the entire state vector.
    * `--state_dropout_mode=elementwise` -> zero each dimension independently.

Custom flags (stripped from argv before draccus parses, so no collision):
    --image_dropout_prob=FLOAT     default 0.3
    --state_dropout_prob=FLOAT     default 0.3
    --state_dropout_mode=STR       'vector' | 'elementwise'  (default 'vector')
    --image_dropout_keys=CSV       camera keys eligible for dropout;
                                   default = auto-detect all `observation.images.*`.

Example:
    python scripts/train_with_feature_dropout.py \
        --dataset.repo_id=my/dataset \
        --policy.type=act \
        --output_dir=outputs/train/act_dropout \
        --image_dropout_prob=0.3 \
        --state_dropout_prob=0.3 \
        --state_dropout_mode=elementwise
"""

import logging
import sys

import torch

import lerobot.scripts.lerobot_train as lt
from lerobot.utils.constants import OBS_IMAGES, OBS_STATE


# --------------------------------------------------------------------------- #
# Custom-flag parsing (kept out of draccus so stock flags are untouched).
# --------------------------------------------------------------------------- #
def _pop_arg(name, cast, default):
    """Remove `--name=value` (or bare `--name`) from sys.argv; return cast value."""
    prefix = f"--{name}="
    for i, arg in enumerate(sys.argv):
        if arg == f"--{name}":  # bare flag -> True
            sys.argv.pop(i)
            return cast("true")
        if arg.startswith(prefix):
            sys.argv.pop(i)
            return cast(arg[len(prefix):])
    return default


class DropoutConfig:
    def __init__(self):
        self.image_prob = _pop_arg("image_dropout_prob", float, 0.3)
        self.state_prob = _pop_arg("state_dropout_prob", float, 0.3)
        self.state_mode = _pop_arg("state_dropout_mode", str, "vector")
        self.image_keys_override = _pop_arg(
            "image_dropout_keys", lambda s: [k for k in s.split(",") if k], None
        )
        if self.state_mode not in ("vector", "elementwise"):
            raise ValueError(
                f"--state_dropout_mode must be 'vector' or 'elementwise', got {self.state_mode!r}"
            )

    @property
    def enabled(self) -> bool:
        return self.image_prob > 0 or self.state_prob > 0

    def summary(self) -> str:
        return (
            f"image_dropout_prob={self.image_prob}, "
            f"state_dropout_prob={self.state_prob}, "
            f"state_dropout_mode={self.state_mode}, "
            f"image_dropout_keys={self.image_keys_override or 'auto (observation.images.*)'}"
        )


# Parse custom flags immediately, before draccus / parser.wrap sees argv.
DROPOUT = DropoutConfig()


def _resolve_image_keys(batch) -> list[str]:
    if DROPOUT.image_keys_override is not None:
        return [k for k in DROPOUT.image_keys_override if k in batch]
    prefix = OBS_IMAGES + "."  # "observation.images."
    return [k for k in batch if isinstance(k, str) and k.startswith(prefix)]


def apply_feature_dropout(batch: dict, cfg: DropoutConfig) -> None:
    """Apply per-sample input-feature dropout in place on a normalized batch."""
    image_keys = _resolve_image_keys(batch)

    # --- Image dropout: at most one camera per sample, always >= 1 image kept.
    if cfg.image_prob > 0 and len(image_keys) >= 1:
        ref = batch[image_keys[0]]
        device = ref.device
        bsz = ref.shape[0]
        drop_sample = torch.rand(bsz, device=device) < cfg.image_prob  # (B,)
        which_cam = torch.randint(len(image_keys), (bsz,), device=device)  # (B,)
        for cam_idx, key in enumerate(image_keys):
            mask = drop_sample & (which_cam == cam_idx)
            if mask.any():
                t = batch[key]
                batch[key] = t * (~mask.view(bsz, *([1] * (t.ndim - 1))))

    # --- Proprioceptive dropout on observation.state.
    if cfg.state_prob > 0 and OBS_STATE in batch:
        state = batch[OBS_STATE]
        bsz = state.shape[0]
        if cfg.state_mode == "vector":
            mask = torch.rand(bsz, device=state.device) < cfg.state_prob
            batch[OBS_STATE] = state * (~mask.view(bsz, *([1] * (state.ndim - 1))))
        else:  # elementwise
            mask = torch.rand_like(state) < cfg.state_prob
            batch[OBS_STATE] = state * (~mask)


# --------------------------------------------------------------------------- #
# Inject dropout by wrapping update_policy (batch arrives post-preprocessor).
# train() calls update_policy via its module global, so patching the module
# attribute is enough -- the real, current train() runs unmodified.
# --------------------------------------------------------------------------- #
_orig_update_policy = lt.update_policy


def _update_policy_with_dropout(train_metrics, policy, batch, *args, **kwargs):
    if DROPOUT.enabled:
        apply_feature_dropout(batch, DROPOUT)
    return _orig_update_policy(train_metrics, policy, batch, *args, **kwargs)


lt.update_policy = _update_policy_with_dropout


def main():
    logging.getLogger(__name__).info("Input-feature dropout: %s", DROPOUT.summary())
    lt.main()


if __name__ == "__main__":
    main()
