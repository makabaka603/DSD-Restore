import argparse
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import PairedRestorationDataset
from losses import DSDRestoreV1Loss
from metrics import batch_psnr, batch_ssim
from models import DSDRestoreV1
from utils.config import load_config
from utils.runtime import ensure_dir, get_device, move_batch_to_device, seed_everything


def build_loaders(cfg: dict) -> tuple[DataLoader, DataLoader]:
    data = cfg["data"]
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
    train_loader = DataLoader(
        train_set,
        batch_size=data["batch_size"],
        shuffle=True,
        num_workers=data.get("num_workers", 0),
        pin_memory=True,
    )
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=0)
    return train_loader, val_loader


@torch.no_grad()
def validate(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    psnr_values, ssim_values = [], []
    for batch in loader:
        batch = move_batch_to_device(batch, device)
        outputs = model(batch["input"])
        psnr_values.append(batch_psnr(outputs["restored"], batch["gt"]))
        ssim_values.append(batch_ssim(outputs["restored"], batch["gt"]))
    return {"psnr": sum(psnr_values) / len(psnr_values), "ssim": sum(ssim_values) / len(ssim_values)}


def train(cfg: dict) -> None:
    seed_everything(cfg["experiment"].get("seed", 42))
    device = get_device(cfg["runtime"].get("device", "auto"))
    output_dir = ensure_dir(cfg["experiment"]["output_dir"])
    ckpt_dir = ensure_dir(output_dir / "checkpoints")
    train_loader, val_loader = build_loaders(cfg)

    model = DSDRestoreV1(**cfg["model"]).to(device)
    criterion = DSDRestoreV1Loss(**cfg["loss"])
    optim = AdamW(model.parameters(), lr=cfg["optimization"]["lr"], weight_decay=cfg["optimization"]["weight_decay"])
    scheduler = CosineAnnealingLR(
        optim,
        T_max=max(1, cfg["optimization"]["epochs"]),
        eta_min=cfg["optimization"].get("min_lr", 1e-6),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=cfg["runtime"].get("amp", False) and device.type == "cuda")
    best_psnr = -1.0

    for epoch in range(1, cfg["optimization"]["epochs"] + 1):
        model.train()
        running = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for batch in pbar:
            batch = move_batch_to_device(batch, device)
            optim.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=scaler.is_enabled()):
                outputs = model(batch["input"])
                losses = criterion(outputs, batch)
            scaler.scale(losses["total"]).backward()
            scaler.step(optim)
            scaler.update()
            running += float(losses["total"].detach().item())
            pbar.set_postfix(loss=f"{running / max(1, pbar.n):.4f}")
        scheduler.step()

        metrics = validate(model, val_loader, device)
        print(f"epoch={epoch} psnr={metrics['psnr']:.3f} ssim={metrics['ssim']:.4f}")
        state = {"model": model.state_dict(), "epoch": epoch, "metrics": metrics, "config": cfg}
        if metrics["psnr"] > best_psnr:
            best_psnr = metrics["psnr"]
            torch.save(state, ckpt_dir / "best.pt")
        if epoch % cfg["runtime"].get("save_every", 10) == 0:
            torch.save(state, ckpt_dir / f"epoch_{epoch:04d}.pt")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_v1_minimal.yaml")
    args = parser.parse_args()
    train(load_config(args.config))


if __name__ == "__main__":
    main()
