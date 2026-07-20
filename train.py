import argparse
import math
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import PairedRestorationDataset, build_balanced_sampler, build_source_dataset
from losses import DSDRestoreV1Loss
from metrics import batch_psnr, batch_ssim
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
) -> dict[str, float | dict[str, dict[str, float]]]:
    model.eval()
    task_values: dict[str, dict[str, list[float]]] = {}
    for batch in loader:
        batch = move_batch_to_device(batch, device)
        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled, dtype=amp_dtype):
            outputs = model(batch["input"])
        task = batch.get("task", ["mixed"])[0]
        values = task_values.setdefault(task, {"psnr": [], "ssim": []})
        values["psnr"].append(batch_psnr(outputs["restored"], batch["gt"]))
        values["ssim"].append(batch_ssim(outputs["restored"], batch["gt"]))
    per_task = {
        task: {
            "psnr": sum(values["psnr"]) / len(values["psnr"]),
            "ssim": sum(values["ssim"]) / len(values["ssim"]),
            "count": len(values["psnr"]),
        }
        for task, values in task_values.items()
    }
    macro_psnr = sum(values["psnr"] for values in per_task.values()) / len(per_task)
    macro_ssim = sum(values["ssim"] for values in per_task.values()) / len(per_task)
    return {"psnr": macro_psnr, "ssim": macro_ssim, "per_task": per_task}


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
    best_psnr: float,
    best_step: int,
    metrics: dict[str, float],
) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optim.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "global_step": global_step,
            "epoch": epoch,
            "best_psnr": best_psnr,
            "best_step": best_step,
            "metrics": metrics,
            "config": cfg,
        },
        path,
    )


def train(cfg: dict, resume: str | None = None) -> None:
    seed_everything(cfg["experiment"].get("seed", 42))
    device = get_device(cfg["runtime"].get("device", "auto"))
    configure_cuda(device, cfg["runtime"].get("cudnn_benchmark", True))
    amp_enabled, amp_dtype = get_amp_settings(
        device,
        cfg["runtime"].get("amp", False),
        cfg["runtime"].get("amp_dtype", "bfloat16"),
    )
    output_dir = ensure_dir(cfg["experiment"]["output_dir"])
    ckpt_dir = ensure_dir(output_dir / "checkpoints")
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
    best_psnr = -1.0
    best_step = 0
    global_step = 0
    epoch = 0
    last_metrics = {"psnr": float("nan"), "ssim": float("nan")}

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
        best_psnr = checkpoint.get("best_psnr", checkpoint.get("metrics", {}).get("psnr", -1.0))
        best_step = checkpoint.get("best_step", global_step)
        last_metrics = checkpoint.get("metrics", last_metrics)
        print(f"Resumed from {resume_path} at iteration {global_step}.")

    print(
        f"device={device} amp={amp_enabled} amp_dtype={amp_dtype} "
        f"batch_size={cfg['data']['batch_size']} max_iters={max_iters}"
    )

    val_every = cfg["runtime"].get("val_every_iters", 2000)
    save_every = cfg["runtime"].get("save_every_iters", 5000)
    early_stop_patience = cfg["runtime"].get("early_stop_patience_iters", 0)
    if val_every <= 0 or save_every <= 0:
        raise ValueError("val_every_iters and save_every_iters must be greater than zero.")
    stop_training = False

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
            pbar.set_postfix(
                iteration=global_step,
                loss=f"{float(losses['total'].detach()):.4f}",
                lr=f"{optim.param_groups[0]['lr']:.2e}",
            )

            should_validate = global_step % val_every == 0 or global_step == max_iters
            if should_validate:
                last_metrics = validate(model, val_loader, device, amp_enabled, amp_dtype)
                print(
                    f"iteration={global_step} macro_psnr={last_metrics['psnr']:.3f} "
                    f"macro_ssim={last_metrics['ssim']:.4f}"
                )
                for task, task_metrics in last_metrics.get("per_task", {}).items():
                    print(
                        f"  {task}: count={task_metrics['count']} "
                        f"psnr={task_metrics['psnr']:.3f} ssim={task_metrics['ssim']:.4f}"
                    )
                if last_metrics["psnr"] > best_psnr:
                    best_psnr = last_metrics["psnr"]
                    best_step = global_step
                    save_checkpoint(
                        ckpt_dir / "best.pt", model, optim, scheduler, scaler, cfg,
                        global_step, epoch, best_psnr, best_step, last_metrics,
                    )
                elif early_stop_patience > 0 and global_step - best_step >= early_stop_patience:
                    print(
                        f"Early stopping at iteration {global_step}: validation PSNR has not "
                        f"improved for {global_step - best_step} iterations."
                    )
                    stop_training = True
                model.train()

            if global_step % save_every == 0 or global_step == max_iters:
                save_checkpoint(
                    ckpt_dir / "latest.pt", model, optim, scheduler, scaler, cfg,
                    global_step, epoch, best_psnr, best_step, last_metrics,
                )
                save_checkpoint(
                    ckpt_dir / f"iter_{global_step:07d}.pt", model, optim, scheduler, scaler, cfg,
                    global_step, epoch, best_psnr, best_step, last_metrics,
                )

            if stop_training:
                save_checkpoint(
                    ckpt_dir / "latest.pt", model, optim, scheduler, scaler, cfg,
                    global_step, epoch, best_psnr, best_step, last_metrics,
                )
                break


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_v1_minimal.yaml")
    parser.add_argument("--resume", default=None, help="checkpoint path used to resume training")
    args = parser.parse_args()
    train(load_config(args.config), resume=args.resume)


if __name__ == "__main__":
    main()
