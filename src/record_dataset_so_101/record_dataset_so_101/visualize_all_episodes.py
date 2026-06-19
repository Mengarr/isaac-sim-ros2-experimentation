#!/usr/bin/env python
"""Visualize *all* (or a subset of) episodes of a LeRobotDataset in a single Rerun session.

The stock `lerobot-dataset-viz` CLI only accepts a single `--episode-index` and spawns a
fresh viewer per call, so browsing 100 episodes means closing/reopening the app 100 times.

This script loads multiple episodes into one Rerun recording and adds an extra `episode`
timeline, so you can scrub between episodes with the timeline slider without restarting.

Examples:

    # All episodes of a local dataset
    python -m record_dataset_so_101.visualize_all_episodes \
        --repo-id me/so101_dataset --root ~/datasets/so101_dataset

    # A subset
    python -m record_dataset_so_101.visualize_all_episodes \
        --repo-id me/so101_dataset --root ~/datasets/so101_dataset --episodes 0 1 2 5

    # A range
    python -m record_dataset_so_101.visualize_all_episodes \
        --repo-id me/so101_dataset --episodes-range 0 20
"""

import argparse
import gc
import logging
import os
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.utils.data
import tqdm

from lerobot.datasets import LeRobotDataset
from lerobot.utils.constants import ACTION, DONE, OBS_STATE, REWARD
from lerobot.utils.utils import init_logging


def ensure_viewer_on_path() -> None:
    """`rr.spawn()` only searches PATH for the `rerun` viewer. When running with the venv's
    python directly (without activating it), the venv's bin/ isn't on PATH, so add the viewer
    binary bundled inside the rerun-sdk wheel."""
    if shutil.which("rerun") is not None:
        return
    import rerun

    cli = Path(rerun.__file__).parent.parent / "rerun_cli"
    if (cli / "rerun").exists():
        os.environ["PATH"] = f"{cli}{os.pathsep}{os.environ.get('PATH', '')}"


def to_hwc_uint8_numpy(chw_float32_torch: torch.Tensor) -> np.ndarray:
    assert chw_float32_torch.dtype == torch.float32
    assert chw_float32_torch.ndim == 3
    c, h, w = chw_float32_torch.shape
    assert c < h and c < w, f"expect channel first images, but instead {chw_float32_torch.shape}"
    return (chw_float32_torch * 255).type(torch.uint8).permute(1, 2, 0).numpy()


def visualize_episodes(
    dataset: LeRobotDataset,
    episodes: list[int],
    batch_size: int = 32,
    num_workers: int = 4,
    display_compressed_images: bool = False,
) -> None:
    repo_id = dataset.repo_id

    import rerun as rr

    # One spawned viewer process; each episode is its own recording that connects to it.
    # All recordings share the application_id, so the viewer's "Recordings" panel lists them
    # together and you can select / replay each episode independently on its own timeline.
    app_id = f"{repo_id}/episodes"

    # Pre-create one RecordingStream per episode. `rr.spawn()` acts on the global default
    # recording, which is disabled until something initializes it; spawning *with* the first
    # recording launches the viewer and connects it, then the rest connect over gRPC.
    streams = {ep: rr.RecordingStream(application_id=app_id, recording_id=f"episode_{ep:04d}") for ep in episodes}
    ensure_viewer_on_path()
    rr.spawn(recording=streams[episodes[0]])
    for ep in episodes[1:]:
        streams[ep].connect_grpc()

    # Manually collect to avoid a hanging blocking flush when num_workers > 0 (see lerobot upstream).
    gc.collect()

    # episode_index -> first global frame index seen, so frame_index restarts at 0 per episode
    # regardless of the dataloader's global indexing.
    first_index_per_ep: dict[int, int] = {}

    def get_rec(ep: int, global_index: int):
        first_index_per_ep.setdefault(ep, global_index)
        return streams[ep], first_index_per_ep[ep]

    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=num_workers,
        batch_size=batch_size,
    )

    logging.info("Logging %d episode(s) to Rerun", len(episodes))
    for batch in tqdm.tqdm(dataloader, total=len(dataloader)):
        for i in range(len(batch["index"])):
            ep = batch["episode_index"][i].item()
            global_index = batch["index"][i].item()
            rec, first_index = get_rec(ep, global_index)
            frame_index = global_index - first_index

            rec.set_time("frame_index", sequence=frame_index)
            rec.set_time("timestamp", timestamp=batch["timestamp"][i].item())

            for key in dataset.meta.camera_keys:
                img = to_hwc_uint8_numpy(batch[key][i])
                img_entity = rr.Image(img).compress() if display_compressed_images else rr.Image(img)
                rec.log(key, img_entity)

            if ACTION in batch:
                for dim_idx, val in enumerate(batch[ACTION][i]):
                    rec.log(f"{ACTION}/{dim_idx}", rr.Scalars(val.item()))

            if OBS_STATE in batch:
                for dim_idx, val in enumerate(batch[OBS_STATE][i]):
                    rec.log(f"state/{dim_idx}", rr.Scalars(val.item()))

            if DONE in batch:
                rec.log(DONE, rr.Scalars(batch[DONE][i].item()))

            if REWARD in batch:
                rec.log(REWARD, rr.Scalars(batch[REWARD][i].item()))

            if "next.success" in batch:
                rec.log("next.success", rr.Scalars(batch["next.success"][i].item()))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo-id", type=str, required=True, help="LeRobotDataset repo id, e.g. `me/so101_dataset`.")
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Local dataset root. Defaults to the HF cache / downloads from the hub.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        nargs="+",
        default=None,
        help="Explicit list of episode indices, e.g. `--episodes 0 1 2 5`. Defaults to all episodes.",
    )
    parser.add_argument(
        "--episodes-range",
        type=int,
        nargs=2,
        metavar=("START", "END"),
        default=None,
        help="Inclusive-exclusive episode range, e.g. `--episodes-range 0 20`.",
    )
    parser.add_argument("--batch-size", type=int, default=32, help="DataLoader batch size.")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader worker processes.")
    parser.add_argument("--tolerance-s", type=float, default=1e-4, help="Timestamp tolerance passed to LeRobotDataset.")
    parser.add_argument(
        "--display-compressed-images",
        action="store_true",
        help="Display compressed images in Rerun instead of uncompressed ones.",
    )
    args = parser.parse_args()

    init_logging()

    # Resolve which episodes to load. Load metadata first to know the total count.
    logging.info("Loading dataset metadata")
    meta_only = LeRobotDataset(args.repo_id, root=args.root, tolerance_s=args.tolerance_s)
    total = meta_only.meta.total_episodes

    if args.episodes is not None and args.episodes_range is not None:
        parser.error("Use either --episodes or --episodes-range, not both.")
    elif args.episodes is not None:
        episodes = sorted(set(args.episodes))
    elif args.episodes_range is not None:
        start, end = args.episodes_range
        episodes = list(range(start, end))
    else:
        episodes = list(range(total))

    invalid = [ep for ep in episodes if ep < 0 or ep >= total]
    if invalid:
        parser.error(f"Episode indices out of range (dataset has {total} episodes): {invalid}")

    logging.info("Loading %d episode(s): %s", len(episodes), episodes)
    dataset = LeRobotDataset(args.repo_id, episodes=episodes, root=args.root, tolerance_s=args.tolerance_s)

    visualize_episodes(
        dataset,
        episodes=episodes,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        display_compressed_images=args.display_compressed_images,
    )


if __name__ == "__main__":
    main()
