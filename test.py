import argparse
import csv
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from datasets import PairedRestorationDataset, build_source_dataset
from metrics import PerceptualMetrics, batch_psnr, batch_ssim
from models import build_model
from utils.config import load_config
from utils.runtime import configure_cuda, get_amp_settings, get_device, move_batch_to_device


METRIC_NAMES = ("psnr", "ssim", "lpips", "dists")


def evaluate_pair(
    pred: torch.Tensor,
    target: torch.Tensor,
    perceptual: PerceptualMetrics,
) -> dict[str, float]:
    perceptual_values = perceptual(pred, target)
    return {
        "psnr": batch_psnr(pred, target),
        "ssim": batch_ssim(pred, target),
        "lpips": perceptual_values["lpips"],
        "dists": perceptual_values["dists"],
    }


def mean_metrics(values: dict[str, list[float]]) -> dict[str, float]:
    return {name: sum(values[name]) / len(values[name]) for name in METRIC_NAMES}


def write_per_image_csv(path: str | Path, rows: list[dict[str, object]]) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "name",
        "source",
        "task",
        "input_psnr",
        "restored_psnr",
        "psnr_gain",
        "input_ssim",
        "restored_ssim",
        "ssim_gain",
        "input_lpips",
        "restored_lpips",
        "lpips_reduction",
        "input_dists",
        "restored_dists",
        "dists_reduction",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return output_path


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_v1_minimal.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--test-input-dir", default=None)
    parser.add_argument("--test-gt-dir", default=None)
    parser.add_argument("--test-metadata", default=None)
    parser.add_argument(
        "--output-csv",
        default=None,
        help="optional path for per-image input/restored metrics and improvements",
    )
    parser.add_argument(
        "--max-samples-per-source",
        type=int,
        default=0,
        help="limit each test source; 0 evaluates every available sample",
    )
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
    if bool(args.test_input_dir) != bool(args.test_gt_dir):
        raise ValueError("--test-input-dir and --test-gt-dir must be provided together")
    if args.test_input_dir:
        dataset = PairedRestorationDataset(
            args.test_input_dir,
            args.test_gt_dir,
            args.test_metadata,
            training=False,
        )
    elif cfg["data"].get("val_sources"):
        test_sources = [dict(source) for source in cfg["data"]["val_sources"]]
        for source in test_sources:
            if args.max_samples_per_source > 0:
                source["max_samples"] = args.max_samples_per_source
            else:
                source.pop("max_samples", None)
        dataset, _ = build_source_dataset(
            test_sources,
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
    model = build_model(cfg["model"]).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    perceptual = PerceptualMetrics(device)
    task_values: dict[str, dict[str, dict[str, list[float]]]] = {}
    per_image_rows: list[dict[str, object]] = []
    for batch in loader:
        batch = move_batch_to_device(batch, device)
        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled, dtype=amp_dtype):
            outputs = model(batch["input"])
        task = batch.get("task", ["mixed"])[0]
        source = batch.get("source", [task])[0]
        name = batch.get("name", [""])[0]
        input_metrics = evaluate_pair(batch["input"], batch["gt"], perceptual)
        restored_metrics = evaluate_pair(outputs["restored"], batch["gt"], perceptual)
        values = task_values.setdefault(
            task,
            {
                "input": {metric: [] for metric in METRIC_NAMES},
                "restored": {metric: [] for metric in METRIC_NAMES},
            },
        )
        for metric in METRIC_NAMES:
            values["input"][metric].append(input_metrics[metric])
            values["restored"][metric].append(restored_metrics[metric])
        per_image_rows.append(
            {
                "name": name,
                "source": source,
                "task": task,
                "input_psnr": input_metrics["psnr"],
                "restored_psnr": restored_metrics["psnr"],
                "psnr_gain": restored_metrics["psnr"] - input_metrics["psnr"],
                "input_ssim": input_metrics["ssim"],
                "restored_ssim": restored_metrics["ssim"],
                "ssim_gain": restored_metrics["ssim"] - input_metrics["ssim"],
                "input_lpips": input_metrics["lpips"],
                "restored_lpips": restored_metrics["lpips"],
                "lpips_reduction": input_metrics["lpips"] - restored_metrics["lpips"],
                "input_dists": input_metrics["dists"],
                "restored_dists": restored_metrics["dists"],
                "dists_reduction": input_metrics["dists"] - restored_metrics["dists"],
            }
        )

    macro_input = {name: 0.0 for name in METRIC_NAMES}
    macro_restored = {name: 0.0 for name in METRIC_NAMES}
    for task, values in task_values.items():
        input_means = mean_metrics(values["input"])
        restored_means = mean_metrics(values["restored"])
        for metric in METRIC_NAMES:
            macro_input[metric] += input_means[metric]
            macro_restored[metric] += restored_means[metric]
        print(
            f"{task}: count={len(values['restored']['psnr'])} "
            f"PSNR={restored_means['psnr']:.3f} SSIM={restored_means['ssim']:.4f} "
            f"LPIPS={restored_means['lpips']:.4f} DISTS={restored_means['dists']:.4f}"
        )
        print(
            f"  Input: PSNR={input_means['psnr']:.3f} SSIM={input_means['ssim']:.4f} "
            f"LPIPS={input_means['lpips']:.4f} DISTS={input_means['dists']:.4f} | "
            f"Gain: PSNR={restored_means['psnr'] - input_means['psnr']:+.3f} "
            f"SSIM={restored_means['ssim'] - input_means['ssim']:+.4f} "
            f"LPIPS reduction={input_means['lpips'] - restored_means['lpips']:+.4f} "
            f"DISTS reduction={input_means['dists'] - restored_means['dists']:+.4f}"
        )
    task_count = len(task_values)
    for metric in METRIC_NAMES:
        restored_value = macro_restored[metric] / task_count
        input_value = macro_input[metric] / task_count
        print(f"Macro {metric.upper()}: {restored_value:.4f}")
        print(f"Input Macro {metric.upper()}: {input_value:.4f}")
        if metric in {"psnr", "ssim"}:
            improvement = restored_value - input_value
            label = "gain"
        else:
            improvement = input_value - restored_value
            label = "reduction"
        print(f"Macro {metric.upper()} {label}: {improvement:+.4f}")

    if args.output_csv:
        output_path = write_per_image_csv(args.output_csv, per_image_rows)
        print(f"Per-image CSV: {output_path.resolve()}")


if __name__ == "__main__":
    main()
