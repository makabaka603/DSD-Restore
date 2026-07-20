import argparse
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from datasets import PairedRestorationDataset, build_source_dataset
from metrics import PerceptualMetrics, batch_psnr, batch_ssim
from models import DSDRestoreV1
from utils.config import load_config
from utils.runtime import configure_cuda, get_amp_settings, get_device, move_batch_to_device


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_v1_minimal.yaml")
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
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
    if cfg["data"].get("val_sources"):
        dataset, _ = build_source_dataset(
            cfg["data"]["val_sources"],
            crop_size=None,
            training=False,
            seed=cfg["experiment"].get("seed", 42),
        )
    else:
        dataset = PairedRestorationDataset(
            cfg["data"]["val_input_dir"],
            cfg["data"]["val_gt_dir"],
            cfg["data"].get("val_metadata"),
            training=False,
        )
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    model = DSDRestoreV1(**cfg["model"]).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    perceptual = PerceptualMetrics(device)
    task_values = {}
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
    macro = {name: 0.0 for name in ("psnr", "ssim", "lpips", "dists")}
    for task, values in task_values.items():
        means = {name: sum(values[name]) / len(values[name]) for name in macro}
        for name in macro:
            macro[name] += means[name]
        print(
            f"{task}: count={len(values['psnr'])} PSNR={means['psnr']:.3f} "
            f"SSIM={means['ssim']:.4f} LPIPS={means['lpips']:.4f} DISTS={means['dists']:.4f}"
        )
    for name, total in macro.items():
        print(f"Macro {name.upper()}: {total / len(task_values):.4f}")


if __name__ == "__main__":
    main()
