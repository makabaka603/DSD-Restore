import argparse
import math
import os
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import PairedRestorationDataset, build_balanced_sampler, build_source_dataset
from losses import DSDRestoreV1Loss
from metrics import PerceptualMetrics, batch_psnr, batch_ssim
from models import DSDRestoreV1
from utils.config import load_config
from utils.runtime import (
    configure_cuda,
    ensure_dir,
    get_amp_settings,
    get_device,
    move_batch_to_device,
    seed_everything,
)


def build_loaders(cfg: dict) -> tuple[DataLoader, DataLoader]:
    data = cfg["data"]
    if data.get("train_sources"):
        seed = cfg["experiment"].get("seed", 42)
        train_set, train_sources = build_source_dataset(
            data["train_sources"],
            crop_size=data.get("crop_size"),
            training=True,
            seed=seed,
        )
        val_set, _ = build_source_dataset(
            data["val_sources"],
            crop_size=None,
            training=False,
            seed=seed,
        )
        samples_per_epoch = data.get("samples_per_epoch", data["batch_size"] * 1000)
        sampler = build_balanced_sampler(
            train_sources,
            data["train_sources"],
            num_samples=samples_per_epoch,
            seed=seed,
        )
        num_workers = data.get("num_workers", 0)
        loader_kwargs = {
            "batch_size": data["batch_size"],
            "sampler": sampler,
            "num_workers": num_workers,
            "pin_memory": True,
            "persistent_workers": data.get("persistent_workers", True) and num_workers > 0,
            "drop_last": True,
        }
        if num_workers > 0:
            loader_kwargs["prefetch_factor"] = data.get("prefetch_factor", 2)
        train_loader = DataLoader(train_set, **loader_kwargs)
        val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
        print("Training sources:")
        for source, source_cfg in zip(train_sources, data["train_sources"]):
            print(
                f"  {source.name}: {len(source)} images, "
                f"probability={float(source_cfg['probability']):.1%}"
            )
        return train_loader, val_loader

    train_set = PairedRestorationDataset(
        data["train_input_dir"],
        data["train_gt_dir"],
        data.get("train_metadata"),
        crop_size=data.get("crop_size"),
        training=True,
    )
    val_set = PairedRestorationDataset(
        data["val_input_dir"],
        data["val_gt_dir"],
        data.get("val_metadata"),
        crop_size=None,
        training=False,
    )
    num_workers = data.get("num_workers", 0)
    loader_kwargs = {
        "batch_size": data["batch_size"],
        "shuffle": True,
        "num_workers": num_workers,
        "pin_memory": True,
        "persistent_workers": data.get("persistent_workers", True) and num_workers > 0,
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = data.get("prefetch_factor", 2)
    train_loader = DataLoader(
        train_set,
        **loader_kwargs,
    )
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
    return train_loader, val_loader


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    perceptual: PerceptualMetrics,
) -> dict[str, float | dict[str, dict[str, float]]]:
    model.eval()
    task_values: dict[str, dict[str, list[float]]] = {}
    for batch in loader:
        batch = move_batch_to_device(batch, device)
        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled, dtype=amp_dtype):
            outputs = model(batch["input"])
        task = batch.get("task", ["mixed"])[0]
        values = task_values.setdefault(task, {"psnr": [], "ssim": [], "lpips": [], "dists": []})
        values["psnr"].append(batch_psnr(outputs["restored"], batch["gt"]))
        values["ssim"].append(batch_ssim(outputs["restored"], batch["gt"]))
        perceptual_values = perceptual(outputs["restored"], batch["gt"])
        values["lpips"].append(perceptual_values["lpips"])
        values["dists"].append(perceptual_values["dists"])
    per_task = {
        task: {
            "psnr": sum(values["psnr"]) / len(values["psnr"]),
            "ssim": sum(values["ssim"]) / len(values["ssim"]),
            "lpips": sum(values["lpips"]) / len(values["lpips"]),
            "dists": sum(values["dists"]) / len(values["dists"]),
            "count": len(values["psnr"]),
        }
        for task, values in task_values.items()
    }
    macro_psnr = sum(values["psnr"] for values in per_task.values()) / len(per_task)
    macro_ssim = sum(values["ssim"] for values in per_task.values()) / len(per_task)
    macro_lpips = sum(values["lpips"] for values in per_task.values()) / len(per_task)
    macro_dists = sum(values["dists"] for values in per_task.values()) / len(per_task)
    return {
        "psnr": macro_psnr,
        "ssim": macro_ssim,
        "lpips": macro_lpips,
        "dists": macro_dists,
        "per_task": per_task,
    }


def prepare_tensorboard_dir(cfg: dict) -> Path:
    """Create /root/tf-logs as a symlink so events live on the data disk."""
    log_dir = Path(cfg.get("log_dir", "/root/tf-logs"))
    storage_dir = Path(cfg.get("log_storage_dir", "/root/autodl-tmp/tf-logs"))
    use_symlink = cfg.get("symlink_log_dir", True)
    if use_symlink:
        storage_dir.mkdir(parents=True, exist_ok=True)
        if log_dir.is_symlink():
            if log_dir.resolve() != storage_dir.resolve():
                raise RuntimeError(f"{log_dir} points to {log_dir.resolve()}, not {storage_dir}")
        elif log_dir.exists():
            raise RuntimeError(
                f"{log_dir} already exists and is not a symlink. Move it to {storage_dir}, "
                f"then run: ln -s {storage_dir} {log_dir}"
            )
        else:
            log_dir.parent.mkdir(parents=True, exist_ok=True)
            os.symlink(storage_dir, log_dir, target_is_directory=True)
    run_dir = log_dir / cfg.get("run_name", "default")
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def is_better(value: float, best: float, metric: str) -> bool:
    return value < best if metric in {"lpips", "dists"} else value > best


def write_metrics(writer: object, prefix: str, metrics: dict, step: int) -> None:
    for name in ("psnr", "ssim", "lpips", "dists"):
        writer.add_scalar(f"{prefix}/{name}", metrics[name], step)


def build_scheduler(optim: AdamW, max_iters: int, warmup_iters: int, min_lr: float) -> LambdaLR:
    base_lr = optim.param_groups[0]["lr"]
    min_ratio = min_lr / base_lr

    def lr_multiplier(step: int) -> float:
        if warmup_iters > 0 and step < warmup_iters:
            return max(1, step + 1) / warmup_iters
        progress = (step - warmup_iters) / max(1, max_iters - warmup_iters)
        progress = min(max(progress, 0.0), 1.0)
        return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optim, lr_lambda=lr_multiplier)


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optim: AdamW,
    scheduler: LambdaLR,
    scaler: torch.amp.GradScaler,
    cfg: dict,
    global_step: int,
    epoch: int,
    best_value: float,
    best_step: int,
    metrics: dict[str, float],
    running_best: dict[str, float] | None = None,
) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optim.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "global_step": global_step,
            "epoch": epoch,
            "best_value": best_value,
            # Kept for compatibility with checkpoints created by older code.
            "best_psnr": best_value if cfg["runtime"].get("selection_metric", "psnr") == "psnr" else None,
            "best_step": best_step,
            "metrics": metrics,
            "running_best": running_best,
            "config": cfg,
        },
        path,
    )


def save_weights(
    path: Path,
    model: torch.nn.Module,
    cfg: dict,
    global_step: int,
    metrics: dict[str, float],
) -> None:
    """Save a compact inference checkpoint without optimizer/scaler states."""
    torch.save(
        {
            "model": model.state_dict(),
            "global_step": global_step,
            "metrics": metrics,
            "config": cfg,
        },
        path,
    )


def train(cfg: dict, resume: str | None = None) -> None:
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError as exc:
        raise ImportError(
            "TensorBoard logging is enabled but tensorboard is not installed. "
            "Run: python -m pip install tensorboard"
        ) from exc

    seed_everything(cfg["experiment"].get("seed", 42))
    torch_home = Path(cfg["runtime"].get("torch_home", "/root/autodl-tmp/torch-cache"))
    torch_home.mkdir(parents=True, exist_ok=True)
    os.environ["TORCH_HOME"] = str(torch_home)
    device = get_device(cfg["runtime"].get("device", "auto"))
    configure_cuda(device, cfg["runtime"].get("cudnn_benchmark", True))
    amp_enabled, amp_dtype = get_amp_settings(
        device,
        cfg["runtime"].get("amp", False),
        cfg["runtime"].get("amp_dtype", "bfloat16"),
    )
    output_dir = ensure_dir(cfg["experiment"]["output_dir"])
    ckpt_dir = ensure_dir(cfg["runtime"].get("checkpoint_dir", output_dir / "checkpoints"))
    tensorboard_cfg = cfg["runtime"].get("tensorboard", {})
    tensorboard_dir = prepare_tensorboard_dir(tensorboard_cfg)
    train_loader, val_loader = build_loaders(cfg)

    metadata_path = cfg["data"].get("train_metadata")
    if not cfg["data"].get("train_sources") and cfg["loss"].get("lambda_cls", 0.0) > 0 and (
        not metadata_path or not Path(metadata_path).exists()
    ):
        raise FileNotFoundError(
            "lambda_cls is greater than zero, but train_metadata is missing. "
            "Provide degradation labels or set loss.lambda_cls to 0.0."
        )

    model = DSDRestoreV1(**cfg["model"]).to(device)
    perceptual = PerceptualMetrics(device)
    criterion = DSDRestoreV1Loss(**cfg["loss"])
    optim = AdamW(model.parameters(), lr=cfg["optimization"]["lr"], weight_decay=cfg["optimization"]["weight_decay"])
    max_iters = cfg["optimization"]["max_iters"]
    scheduler = build_scheduler(
        optim,
        max_iters=max_iters,
        warmup_iters=cfg["optimization"].get("warmup_iters", 0),
        min_lr=cfg["optimization"].get("min_lr", 1e-6),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled and amp_dtype == torch.float16)
    selection_metric = cfg["runtime"].get("selection_metric", "psnr").lower()
    if selection_metric not in {"psnr", "ssim", "lpips", "dists"}:
        raise ValueError("selection_metric must be one of: psnr, ssim, lpips, dists")
    best_value = math.inf if selection_metric in {"lpips", "dists"} else -math.inf
    best_step = 0
    global_step = 0
    epoch = 0
    last_metrics = {name: float("nan") for name in ("psnr", "ssim", "lpips", "dists")}
    running_best = {"psnr": -math.inf, "ssim": -math.inf, "lpips": math.inf, "dists": math.inf}

    resume_path = resume or cfg["runtime"].get("resume")
    if resume_path:
        checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optim.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        if checkpoint.get("scaler"):
            scaler.load_state_dict(checkpoint["scaler"])
        global_step = checkpoint.get("global_step", 0)
        epoch = checkpoint.get("epoch", 0)
        best_value = checkpoint.get("best_value")
        if best_value is None:
            best_value = checkpoint.get("best_psnr", checkpoint.get("metrics", {}).get(selection_metric, best_value))
        best_step = checkpoint.get("best_step", global_step)
        last_metrics = checkpoint.get("metrics", last_metrics)
        running_best.update(checkpoint.get("running_best") or {})
        print(f"Resumed from {resume_path} at iteration {global_step}.")

    # purge_step prevents a resumed run from displaying stale duplicate points
    # at and after the first newly written iteration.
    writer = SummaryWriter(
        tensorboard_dir,
        purge_step=global_step + 1 if resume_path else None,
    )

    print(
        f"device={device} amp={amp_enabled} amp_dtype={amp_dtype} "
        f"batch_size={cfg['data']['batch_size']} max_iters={max_iters}"
    )

    val_every = cfg["runtime"].get("val_every_iters", 2000)
    save_every = cfg["runtime"].get("save_every_iters", 5000)
    window_every = cfg["runtime"].get("best_window_iters", 100)
    train_metrics_cfg = cfg["runtime"].get("train_metrics", {})
    psnr_ssim_every = train_metrics_cfg.get("psnr_ssim_every_iters", 1)
    perceptual_every = train_metrics_cfg.get("perceptual_every_iters", 100)
    perceptual_samples = train_metrics_cfg.get("perceptual_batch_samples", 4)
    early_stop_patience = cfg["runtime"].get("early_stop_patience_iters", 0)
    if min(val_every, save_every, window_every, psnr_ssim_every, perceptual_every, perceptual_samples) <= 0:
        raise ValueError("validation, save, metric, sample, and best-window values must be greater than zero.")
    if val_every > window_every or window_every % val_every != 0:
        raise ValueError(
            "best_window_iters must be an exact multiple of val_every_iters so every "
            "checkpoint window contains comparable validation measurements."
        )
    stop_training = False
    window_best = math.inf if selection_metric in {"lpips", "dists"} else -math.inf
    active_window_start = ((global_step // window_every) * window_every) + 1

    while global_step < max_iters and not stop_training:
        epoch += 1
        model.train()
        remaining = max_iters - global_step
        pbar = tqdm(train_loader, total=min(len(train_loader), remaining), desc=f"Epoch {epoch}")
        for batch in pbar:
            if global_step >= max_iters:
                break
            batch = move_batch_to_device(batch, device)
            optim.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled, dtype=amp_dtype):
                outputs = model(batch["input"])
                losses = criterion(outputs, batch)
            scaler.scale(losses["total"]).backward()
            scaler.step(optim)
            scaler.update()
            scheduler.step()
            global_step += 1
            restored = outputs["restored"].detach()
            writer.add_scalar("train/loss", float(losses["total"].detach()), global_step)
            writer.add_scalar("train/learning_rate", optim.param_groups[0]["lr"], global_step)

            train_metrics = {}
            if global_step % psnr_ssim_every == 0:
                train_metrics.update(
                    psnr=batch_psnr(restored, batch["gt"]),
                    ssim=batch_ssim(restored, batch["gt"]),
                )
            if global_step % perceptual_every == 0:
                sample_count = min(perceptual_samples, restored.shape[0])
                train_metrics.update(
                    perceptual(restored[:sample_count], batch["gt"][:sample_count])
                )
            for name, value in train_metrics.items():
                writer.add_scalar(f"train/current/{name}", value, global_step)
                if is_better(value, running_best[name], name):
                    running_best[name] = value
                writer.add_scalar(f"train/best/{name}", running_best[name], global_step)
            pbar.set_postfix(
                iteration=global_step,
                loss=f"{float(losses['total'].detach()):.4f}",
                lr=f"{optim.param_groups[0]['lr']:.2e}",
            )

            should_validate = global_step % val_every == 0 or global_step == max_iters
            if should_validate:
                last_metrics = validate(model, val_loader, device, amp_enabled, amp_dtype, perceptual)
                write_metrics(writer, "val", last_metrics, global_step)
                for task, task_metrics in last_metrics.get("per_task", {}).items():
                    write_metrics(writer, f"val_by_task/{task}", task_metrics, global_step)
                print(
                    f"iteration={global_step} macro_psnr={last_metrics['psnr']:.3f} "
                    f"macro_ssim={last_metrics['ssim']:.4f} "
                    f"macro_lpips={last_metrics['lpips']:.4f} macro_dists={last_metrics['dists']:.4f}"
                )
                for task, task_metrics in last_metrics.get("per_task", {}).items():
                    print(
                        f"  {task}: count={task_metrics['count']} "
                        f"psnr={task_metrics['psnr']:.3f} ssim={task_metrics['ssim']:.4f} "
                        f"lpips={task_metrics['lpips']:.4f} dists={task_metrics['dists']:.4f}"
                    )
                if is_better(last_metrics[selection_metric], best_value, selection_metric):
                    best_value = last_metrics[selection_metric]
                    best_step = global_step
                    save_weights(ckpt_dir / "best.pth", model, cfg, global_step, last_metrics)
                elif early_stop_patience > 0 and global_step - best_step >= early_stop_patience:
                    print(
                        f"Early stopping at iteration {global_step}: validation {selection_metric} has not "
                        f"improved for {global_step - best_step} iterations."
                    )
                    stop_training = True

                # Keep overwriting this window's file only while its validation
                # selection metric improves. Thus val_every_iters may be smaller
                # than best_window_iters without producing redundant snapshots.
                window_start = ((global_step - 1) // window_every) * window_every + 1
                window_end = window_start + window_every - 1
                if window_start != active_window_start:
                    active_window_start = window_start
                    window_best = math.inf if selection_metric in {"lpips", "dists"} else -math.inf
                if is_better(last_metrics[selection_metric], window_best, selection_metric):
                    window_best = last_metrics[selection_metric]
                    save_weights(
                        ckpt_dir / f"window_{window_start:07d}_{window_end:07d}_best.pth",
                        model, cfg, global_step, last_metrics,
                    )
                model.train()

            if global_step % save_every == 0 or global_step == max_iters:
                save_checkpoint(
                    ckpt_dir / "latest.pth", model, optim, scheduler, scaler, cfg,
                    global_step, epoch, best_value, best_step, last_metrics, running_best,
                )
                save_checkpoint(
                    ckpt_dir / f"iter_{global_step:07d}.pth", model, optim, scheduler, scaler, cfg,
                    global_step, epoch, best_value, best_step, last_metrics, running_best,
                )

            if stop_training:
                save_checkpoint(
                    ckpt_dir / "latest.pth", model, optim, scheduler, scaler, cfg,
                    global_step, epoch, best_value, best_step, last_metrics, running_best,
                )
                break
    writer.flush()
    writer.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_v1_minimal.yaml")
    parser.add_argument("--resume", default=None, help="checkpoint path used to resume training")
    args = parser.parse_args()
    train(load_config(args.config), resume=args.resume)


if __name__ == "__main__":
    main()
