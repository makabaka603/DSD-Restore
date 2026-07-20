import argparse

import torch
from torch.utils.data import DataLoader

from datasets import PairedRestorationDataset, build_source_dataset
from metrics import batch_psnr, batch_ssim
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
    task_values = {}
    for batch in loader:
        batch = move_batch_to_device(batch, device)
        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled, dtype=amp_dtype):
            outputs = model(batch["input"])
        task = batch.get("task", ["mixed"])[0]
        values = task_values.setdefault(task, {"psnr": [], "ssim": []})
        values["psnr"].append(batch_psnr(outputs["restored"], batch["gt"]))
        values["ssim"].append(batch_ssim(outputs["restored"], batch["gt"]))
    macro_psnr, macro_ssim = 0.0, 0.0
    for task, values in task_values.items():
        psnr = sum(values["psnr"]) / len(values["psnr"])
        ssim = sum(values["ssim"]) / len(values["ssim"])
        macro_psnr += psnr
        macro_ssim += ssim
        print(f"{task}: count={len(values['psnr'])} PSNR={psnr:.3f} SSIM={ssim:.4f}")
    print(f"Macro PSNR: {macro_psnr / len(task_values):.3f}")
    print(f"Macro SSIM: {macro_ssim / len(task_values):.4f}")


if __name__ == "__main__":
    main()
