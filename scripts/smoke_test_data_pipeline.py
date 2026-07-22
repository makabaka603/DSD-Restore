import argparse
import sys
from collections import Counter
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from losses import DSDRestoreV1Loss
from models import DSDRestoreV1
from train import build_loaders
from utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_v1_stage2.yaml")
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()
    cfg = load_config(args.config)
    cfg["data"]["batch_size"] = 4
    cfg["data"]["num_workers"] = args.num_workers
    cfg["data"]["persistent_workers"] = args.num_workers > 0
    cfg["data"]["samples_per_epoch"] = 40
    train_loader, _ = build_loaders(cfg)

    source_counts = Counter()
    task_counts = Counter()
    first_batch = None
    for batch in train_loader:
        if first_batch is None:
            first_batch = batch
        source_counts.update(batch["source"])
        task_counts.update(batch["task"])
    if first_batch is None:
        raise RuntimeError("The training loader returned no batches")
    if first_batch["input"].shape != first_batch["gt"].shape:
        raise ValueError("Input and GT batch shapes differ")
    if first_batch["dense_label"].shape[1] != 5 or first_batch["sparse_label"].shape[1] != 4:
        raise ValueError("Unexpected degradation label dimensions")
    stage = cfg.get("stage", {}).get("name")
    has_single = any("+" not in task for task in task_counts)
    has_composite = any("+" in task for task in task_counts)
    if stage == "stage1" and has_composite:
        raise ValueError("Stage 1 sampled a composite degradation")
    if stage == "stage3" and has_single:
        raise ValueError("Stage 3 sampled a single degradation")
    if stage not in {"stage1", "stage3"} and not has_composite:
        raise ValueError("Joint training sampled no composite degradation")
    for key in ("dense_label", "sparse_label"):
        labels = first_batch[key]
        if float(labels.min()) < 0.0 or float(labels.max()) > 1.0:
            raise ValueError(f"{key} contains values outside [0, 1]")

    # Keep the backward pass quick while still using a real mixed-source batch.
    model = DSDRestoreV1(base_channels=16, token_dim=32)
    small_batch = {
        "input": first_batch["input"][:, :, :64, :64],
        "gt": first_batch["gt"][:, :, :64, :64],
        "dense_label": first_batch["dense_label"],
        "sparse_label": first_batch["sparse_label"],
    }
    outputs = model(small_batch["input"])
    losses = DSDRestoreV1Loss()(outputs, small_batch)
    losses["total"].backward()

    print("Data pipeline smoke test passed")
    print(f"config: {args.config} stage: {stage or 'unlabeled'}")
    print(f"batch shape: {tuple(first_batch['input'].shape)}")
    print(f"sources observed in 40 draws: {dict(sorted(source_counts.items()))}")
    print(f"tasks observed in 40 draws: {dict(sorted(task_counts.items()))}")
    print(f"loss: {float(losses['total'].detach()):.4f}")


if __name__ == "__main__":
    main()
