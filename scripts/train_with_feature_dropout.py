#!/usr/bin/env python
"""Train a LeRobot policy with input-feature dropout.

This is a thin wrapper around `lerobot.scripts.lerobot_train.train` that adds
*input feature dropout* as a regularizer. It is a drop-in replacement for the
`lerobot-train` CLI: every standard training flag still works exactly the same,
because the dropout flags are stripped from `sys.argv` *before* draccus parses
the `TrainPipelineConfig`.

Dropout semantics (applied per training step, after the preprocessor runs, so
all features are already normalized -- a "zeroed" feature equals the dataset
mean of that feature, i.e. the least-informative imputation):

  Images (cameras):
    * Per sample, with probability `--image_dropout_prob`, exactly ONE camera is
      chosen uniformly at random and blacked out (zeroed).
    * At most one camera is ever dropped per sample, so the policy is always fed
      at least one image.

  Proprioception (observation.state):
    * Per sample, dropped with probability `--state_dropout_prob`.
    * `--state_dropout_mode=vector`      -> zero the entire state vector.
    * `--state_dropout_mode=elementwise` -> zero each dimension independently.

Custom flags (all optional):
    --image_dropout_prob=FLOAT     default 0.3
    --state_dropout_prob=FLOAT     default 0.3
    --state_dropout_mode=STR       'vector' | 'elementwise'  (default 'vector')
    --image_dropout_keys=CSV       comma-separated camera keys to consider for
                                   dropout; default = all dataset cameras.

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
import time
from contextlib import nullcontext  # noqa: F401 (kept for parity with upstream)
from pprint import pformat
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from accelerate import Accelerator

import torch
from termcolor import colored
from tqdm import tqdm

from lerobot.common.train_utils import (
    get_step_checkpoint_dir,
    get_step_identifier,
    load_training_state,
    save_checkpoint,
    update_last_checkpoint,
)
from lerobot.common.wandb_utils import WandBLogger
from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets import EpisodeAwareSampler, make_dataset
from lerobot.envs import close_envs, make_env, make_env_pre_post_processors
from lerobot.optim.factory import make_optimizer_and_scheduler
from lerobot.policies import make_policy, make_pre_post_processors
from lerobot.rewards import make_reward_pre_post_processors
from lerobot.utils.collate import lerobot_collate_fn
from lerobot.utils.constants import OBS_STATE
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.logging_utils import AverageMeter, MetricsTracker
from lerobot.utils.random_utils import set_seed
from lerobot.utils.utils import (
    cycle,
    format_big_number,
    init_logging,
    inside_slurm,
)

# Reuse the upstream training-step implementation verbatim.
from lerobot.scripts.lerobot_train import update_policy
from lerobot.scripts.lerobot_eval import eval_policy_all


# --------------------------------------------------------------------------- #
# Custom-flag parsing (kept out of draccus so stock flags are untouched).
# --------------------------------------------------------------------------- #
def _pop_arg(name: str, cast, default):
    """Remove `--name=value` from sys.argv and return the cast value (or default)."""
    prefix = f"--{name}="
    for i, arg in enumerate(sys.argv):
        if arg == prefix.rstrip("="):  # `--name` with no value -> treat as flag/True
            sys.argv.pop(i)
            return cast("true")
        if arg.startswith(prefix):
            sys.argv.pop(i)
            return cast(arg[len(prefix):])
    return default


class DropoutConfig:
    """Holds the parsed input-feature-dropout settings."""

    def __init__(self):
        self.image_prob = _pop_arg("image_dropout_prob", float, 0.3)
        self.state_prob = _pop_arg("state_dropout_prob", float, 0.3)
        self.state_mode = _pop_arg("state_dropout_mode", str, "vector")
        keys = _pop_arg("image_dropout_keys", lambda s: [k for k in s.split(",") if k], None)
        self.image_keys_override = keys

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
            f"image_dropout_keys={self.image_keys_override or 'ALL'}"
        )


# Parse the custom flags immediately, before parser.wrap() / draccus sees argv.
DROPOUT = DropoutConfig()


def apply_feature_dropout(batch: dict, image_keys: list[str], cfg: DropoutConfig) -> None:
    """Apply per-sample input-feature dropout in place on a (normalized) batch.

    Args:
        batch: The post-preprocessor batch dict (tensors already normalized).
        image_keys: Camera keys present in this batch eligible for dropout.
        cfg: The dropout configuration.
    """
    # --- Image dropout: at most one camera per sample, always >= 1 image kept.
    if cfg.image_prob > 0 and len(image_keys) >= 1:
        ref = batch[image_keys[0]]
        device = ref.device
        bsz = ref.shape[0]

        # Which samples drop a camera at all.
        drop_sample = torch.rand(bsz, device=device) < cfg.image_prob
        # For those samples, which camera index (0..K-1) to drop.
        which_cam = torch.randint(len(image_keys), (bsz,), device=device)

        for cam_idx, key in enumerate(image_keys):
            # Sample is masked iff it was selected to drop AND this camera is the chosen one.
            mask = drop_sample & (which_cam == cam_idx)  # (B,)
            if mask.any():
                t = batch[key]
                view = mask.view(bsz, *([1] * (t.ndim - 1)))
                batch[key] = t * (~view)

    # --- Proprioceptive dropout on observation.state.
    if cfg.state_prob > 0 and OBS_STATE in batch:
        state = batch[OBS_STATE]
        device = state.device
        bsz = state.shape[0]

        if cfg.state_mode == "vector":
            # Per sample: zero the whole state vector with prob state_prob.
            mask = torch.rand(bsz, device=device) < cfg.state_prob  # (B,)
            view = mask.view(bsz, *([1] * (state.ndim - 1)))
            batch[OBS_STATE] = state * (~view)
        else:  # elementwise
            # Per sample, per dimension: zero independently with prob state_prob.
            mask = torch.rand_like(state) < cfg.state_prob
            batch[OBS_STATE] = state * (~mask)


@parser.wrap()
def train(cfg: TrainPipelineConfig, accelerator: "Accelerator | None" = None):
    """Train a policy with input-feature dropout. Mirrors lerobot_train.train()."""
    from lerobot.utils.import_utils import require_package

    require_package("accelerate", extra="training")
    from accelerate import Accelerator

    cfg.validate()

    if accelerator is None:
        from accelerate.utils import DistributedDataParallelKwargs

        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        force_cpu = cfg.trainable_config.device == "cpu"
        accelerator = Accelerator(
            step_scheduler_with_optimizer=False,
            kwargs_handlers=[ddp_kwargs],
            cpu=force_cpu,
        )

    init_logging(accelerator=accelerator)
    is_main_process = accelerator.is_main_process

    if is_main_process:
        logging.info(pformat(cfg.to_dict()))
        logging.info(colored(f"Input-feature dropout: {DROPOUT.summary()}", "cyan", attrs=["bold"]))

    if cfg.wandb.enable and cfg.wandb.project and is_main_process:
        wandb_logger = WandBLogger(cfg)
    else:
        wandb_logger = None
        if is_main_process:
            logging.info(colored("Logs will be saved locally.", "yellow", attrs=["bold"]))

    if cfg.seed is not None:
        set_seed(cfg.seed, accelerator=accelerator)

    device = accelerator.device
    if cfg.cudnn_deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    if is_main_process:
        logging.info("Creating dataset")
        dataset = make_dataset(cfg)
    accelerator.wait_for_everyone()
    if not is_main_process:
        dataset = make_dataset(cfg)

    eval_env = None
    if cfg.eval_freq > 0 and cfg.env is not None and is_main_process:
        logging.info("Creating env")
        eval_env = make_env(cfg.env, n_envs=cfg.eval.batch_size, use_async_envs=cfg.eval.use_async_envs)

    if cfg.is_reward_model_training:
        if is_main_process:
            logging.info("Creating reward model")
        from lerobot.rewards import make_reward_model

        policy = make_reward_model(
            cfg=cfg.reward_model,
            dataset_stats=dataset.meta.stats,
            dataset_meta=dataset.meta,
        )
        if not policy.is_trainable:
            raise ValueError(
                f"Reward model '{policy.name}' is zero-shot and cannot be trained via lerobot-train."
            )
    else:
        if is_main_process:
            logging.info("Creating policy")
        policy = make_policy(
            cfg=cfg.policy,
            ds_meta=dataset.meta,
            rename_map=cfg.rename_map,
        )

    if cfg.peft is not None:
        if cfg.is_reward_model_training:
            raise ValueError("PEFT is only supported for policy training. ")
        import dataclasses

        from peft import PeftModel

        if isinstance(policy, PeftModel):
            logging.info("PEFT adapter already loaded from checkpoint, skipping wrap_with_peft.")
        else:
            logging.info("Using PEFT! Wrapping model.")
            peft_cli_overrides = dataclasses.asdict(cfg.peft)
            policy = policy.wrap_with_peft(peft_cli_overrides=peft_cli_overrides)

    accelerator.wait_for_everyone()

    active_cfg = cfg.trainable_config
    processor_pretrained_path = active_cfg.pretrained_path

    processor_kwargs = {}
    if (processor_pretrained_path and not cfg.resume) or not processor_pretrained_path:
        processor_kwargs["dataset_stats"] = dataset.meta.stats
    if cfg.is_reward_model_training:
        processor_kwargs["dataset_meta"] = dataset.meta

    if not cfg.is_reward_model_training and processor_pretrained_path is not None:
        preprocessor_overrides = {
            "device_processor": {"device": device.type},
            "normalizer_processor": {
                "stats": dataset.meta.stats,
                "features": {**policy.config.input_features, **policy.config.output_features},
                "norm_map": policy.config.normalization_mapping,
            },
            "rename_observations_processor": {"rename_map": cfg.rename_map},
        }
        postprocessor_overrides = {
            "unnormalizer_processor": {
                "stats": dataset.meta.stats,
                "features": policy.config.output_features,
                "norm_map": policy.config.normalization_mapping,
            },
        }
        if getattr(active_cfg, "use_relative_actions", False):
            preprocessor_overrides["relative_actions_processor"] = {
                "enabled": True,
                "exclude_joints": getattr(active_cfg, "relative_exclude_joints", []),
                "action_names": getattr(active_cfg, "action_feature_names", None),
            }
            postprocessor_overrides["absolute_actions_processor"] = {"enabled": True}
        processor_kwargs["preprocessor_overrides"] = preprocessor_overrides
        processor_kwargs["postprocessor_overrides"] = postprocessor_overrides

    if cfg.is_reward_model_training:
        preprocessor, postprocessor = make_reward_pre_post_processors(
            cfg.reward_model,
            **processor_kwargs,
        )
    else:
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=cfg.policy,
            pretrained_path=processor_pretrained_path,
            **processor_kwargs,
        )

    if is_main_process:
        logging.info("Creating optimizer and scheduler")
    optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, policy)

    sample_weighter = None
    if cfg.sample_weighting is not None:
        from lerobot.utils.sample_weighting import make_sample_weighter

        if is_main_process:
            logging.info(f"Creating sample weighter: {cfg.sample_weighting.type}")
        sample_weighter = make_sample_weighter(
            cfg.sample_weighting,
            policy,
            device,
            dataset_root=cfg.dataset.root,
            dataset_repo_id=cfg.dataset.repo_id,
        )

    step = 0
    if cfg.resume:
        step, optimizer, lr_scheduler = load_training_state(cfg.checkpoint_path, optimizer, lr_scheduler)

    num_learnable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    num_total_params = sum(p.numel() for p in policy.parameters())

    # Resolve which camera keys are eligible for image dropout.
    image_keys = DROPOUT.image_keys_override or list(dataset.meta.camera_keys)

    if is_main_process:
        logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")
        logging.info(colored(f"Image dropout keys: {image_keys}", "cyan"))
        if cfg.env is not None:
            logging.info(f"{cfg.env.task=}")
            logging.info("Creating environment processors")
            env_preprocessor, env_postprocessor = make_env_pre_post_processors(
                env_cfg=cfg.env, policy_cfg=cfg.policy
            )
        logging.info(f"{cfg.steps=} ({format_big_number(cfg.steps)})")
        logging.info(f"{dataset.num_frames=} ({format_big_number(dataset.num_frames)})")
        logging.info(f"{dataset.num_episodes=}")
        num_processes = accelerator.num_processes
        effective_bs = cfg.batch_size * num_processes
        logging.info(f"Effective batch size: {cfg.batch_size} x {num_processes} = {effective_bs}")
        logging.info(f"{num_learnable_params=} ({format_big_number(num_learnable_params)})")
        logging.info(f"{num_total_params=} ({format_big_number(num_total_params)})")

    if hasattr(active_cfg, "drop_n_last_frames"):
        shuffle = False
        sampler = EpisodeAwareSampler(
            dataset.meta.episodes["dataset_from_index"],
            dataset.meta.episodes["dataset_to_index"],
            episode_indices_to_use=dataset.episodes,
            drop_n_last_frames=active_cfg.drop_n_last_frames,
            shuffle=True,
        )
    else:
        shuffle = True
        sampler = None

    collate_fn = lerobot_collate_fn if dataset.meta.has_language_columns else None
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.num_workers,
        batch_size=cfg.batch_size,
        shuffle=shuffle and not cfg.dataset.streaming,
        sampler=sampler,
        pin_memory=device.type == "cuda",
        drop_last=False,
        collate_fn=collate_fn,
        prefetch_factor=cfg.prefetch_factor if cfg.num_workers > 0 else None,
        persistent_workers=cfg.persistent_workers and cfg.num_workers > 0,
    )

    accelerator.wait_for_everyone()
    policy, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        policy, optimizer, dataloader, lr_scheduler
    )
    dl_iter = cycle(dataloader)
    policy.train()

    train_metrics = {
        "loss": AverageMeter("loss", ":.3f"),
        "grad_norm": AverageMeter("grdn", ":.3f"),
        "lr": AverageMeter("lr", ":0.1e"),
        "update_s": AverageMeter("updt_s", ":.3f"),
        "dataloading_s": AverageMeter("data_s", ":.3f"),
    }

    effective_batch_size = cfg.batch_size * accelerator.num_processes
    train_tracker = MetricsTracker(
        cfg.batch_size,
        dataset.num_frames,
        dataset.num_episodes,
        train_metrics,
        initial_step=step,
        accelerator=accelerator,
    )

    if is_main_process:
        progbar = tqdm(
            total=cfg.steps - step,
            desc="Training",
            unit="step",
            disable=inside_slurm(),
            position=0,
            leave=True,
        )
        logging.info(
            f"Start offline training on a fixed dataset, with effective batch size: {effective_batch_size}"
        )

    for _ in range(step, cfg.steps):
        start_time = time.perf_counter()
        batch = next(dl_iter)
        for cam_key in dataset.meta.camera_keys:
            if cam_key in batch and batch[cam_key].dtype == torch.uint8:
                batch[cam_key] = batch[cam_key].to(dtype=torch.float32) / 255.0
        batch = preprocessor(batch)

        # ---- Input-feature dropout (the only addition vs. upstream train()).
        if DROPOUT.enabled:
            apply_feature_dropout(batch, [k for k in image_keys if k in batch], DROPOUT)

        train_tracker.dataloading_s = time.perf_counter() - start_time

        train_tracker, output_dict = update_policy(
            train_tracker,
            policy,
            batch,
            optimizer,
            cfg.optimizer.grad_clip_norm,
            accelerator=accelerator,
            lr_scheduler=lr_scheduler,
            sample_weighter=sample_weighter,
        )

        step += 1
        if is_main_process:
            progbar.update(1)
        train_tracker.step()
        is_log_step = cfg.log_freq > 0 and step % cfg.log_freq == 0 and is_main_process
        is_saving_step = step % cfg.save_freq == 0 or step == cfg.steps
        is_eval_step = cfg.eval_freq > 0 and step % cfg.eval_freq == 0

        if is_log_step:
            logging.info(train_tracker)
            if wandb_logger:
                wandb_log_dict = train_tracker.to_dict()
                if output_dict:
                    wandb_log_dict.update(output_dict)
                if sample_weighter is not None:
                    weighter_stats = sample_weighter.get_stats()
                    wandb_log_dict.update({f"sample_weighting/{k}": v for k, v in weighter_stats.items()})
                wandb_logger.log_dict(wandb_log_dict, step)
            train_tracker.reset_averages()

        if cfg.save_checkpoint and is_saving_step:
            if is_main_process:
                logging.info(f"Checkpoint policy after step {step}")
                checkpoint_dir = get_step_checkpoint_dir(cfg.output_dir, cfg.steps, step)
                save_checkpoint(
                    checkpoint_dir=checkpoint_dir,
                    step=step,
                    cfg=cfg,
                    policy=accelerator.unwrap_model(policy),
                    optimizer=optimizer,
                    scheduler=lr_scheduler,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                )
                update_last_checkpoint(checkpoint_dir)
                if wandb_logger:
                    wandb_logger.log_policy(checkpoint_dir)
            accelerator.wait_for_everyone()

        if cfg.env and is_eval_step:
            if is_main_process:
                step_id = get_step_identifier(step, cfg.steps)
                logging.info(f"Eval policy at step {step}")
                with torch.no_grad(), accelerator.autocast():
                    eval_info = eval_policy_all(
                        envs=eval_env,
                        policy=accelerator.unwrap_model(policy),
                        env_preprocessor=env_preprocessor,
                        env_postprocessor=env_postprocessor,
                        preprocessor=preprocessor,
                        postprocessor=postprocessor,
                        n_episodes=cfg.eval.n_episodes,
                        videos_dir=cfg.output_dir / "eval" / f"videos_step_{step_id}",
                        max_episodes_rendered=4,
                        start_seed=cfg.seed,
                        max_parallel_tasks=cfg.env.max_parallel_tasks,
                    )
                aggregated = eval_info["overall"]
                for suite, suite_info in eval_info.items():
                    logging.info("Suite %s aggregated: %s", suite, suite_info)
                eval_metrics = {
                    "avg_sum_reward": AverageMeter("∑rwrd", ":.3f"),
                    "pc_success": AverageMeter("success", ":.1f"),
                    "eval_s": AverageMeter("eval_s", ":.3f"),
                }
                eval_tracker = MetricsTracker(
                    cfg.batch_size,
                    dataset.num_frames,
                    dataset.num_episodes,
                    eval_metrics,
                    initial_step=step,
                    accelerator=accelerator,
                )
                eval_tracker.eval_s = aggregated.pop("eval_s")
                eval_tracker.avg_sum_reward = aggregated.pop("avg_sum_reward")
                eval_tracker.pc_success = aggregated.pop("pc_success")
                if wandb_logger:
                    wandb_log_dict = {**eval_tracker.to_dict(), **eval_info}
                    wandb_logger.log_dict(wandb_log_dict, step, mode="eval")
                    wandb_logger.log_video(eval_info["overall"]["video_paths"][0], step, mode="eval")
            accelerator.wait_for_everyone()

    if is_main_process:
        progbar.close()
    if eval_env:
        close_envs(eval_env)

    if is_main_process:
        logging.info("End of training")
        if getattr(active_cfg, "push_to_hub", False):
            unwrapped_model = accelerator.unwrap_model(policy)
            if not cfg.is_reward_model_training and cfg.policy.use_peft:
                unwrapped_model.push_model_to_hub(cfg, peft_model=unwrapped_model)
            else:
                unwrapped_model.push_model_to_hub(cfg)
            preprocessor.push_to_hub(active_cfg.repo_id)
            postprocessor.push_to_hub(active_cfg.repo_id)

    accelerator.wait_for_everyone()
    accelerator.end_training()


def main():
    register_third_party_plugins()
    train()


if __name__ == "__main__":
    main()
