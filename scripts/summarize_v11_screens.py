import argparse
import csv
from pathlib import Path

import torch


SCREEN_RUNS = (
    "v11_screen_a0_control",
    "v11_screen_a1_aux_off",
    "v11_screen_a2_dual_gate",
    "v11_screen_a3_weighted_tokens",
    "v11_screen_a4_multilabel_proto",
    "v11_screen_a5_full",
    "v11b_screen_b1_dual_gate_migrated",
    "v11b_screen_b2_full_migrated",
    "v2lite_screen_c1_multiscale",
    "screen_d0_triple_balance",
    "screen_d1_shared_capacity",
    "screen_v2_factor_spatial_routing",
)


def mean_task_psnr(per_task: dict, predicate) -> float | None:
    values = [
        float(metrics["psnr"])
        for task, metrics in per_task.items()
        if predicate(task) and "psnr" in metrics
    ]
    return sum(values) / len(values) if values else None


def load_row(checkpoint_path: Path, name: str) -> dict[str, object]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    metrics = checkpoint.get("metrics", {})
    per_task = metrics.get("per_task", {})
    return {
        "run": name,
        "step": checkpoint.get("global_step", ""),
        "psnr": metrics.get("psnr", ""),
        "ssim": metrics.get("ssim", ""),
        "lpips": metrics.get("lpips", ""),
        "dists": metrics.get("dists", ""),
        "lowlight_psnr": mean_task_psnr(
            per_task, lambda task: "lowlight" in task
        ),
        "triple_psnr": mean_task_psnr(
            per_task, lambda task: task.count("+") >= 2
        ),
        "checkpoint": str(checkpoint_path),
    }


def format_value(value: object, digits: int = 4) -> str:
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize best validation checkpoints from V1.1 screens."
    )
    parser.add_argument(
        "--checkpoint-root",
        default="/root/autodl-tmp/DSD-Restor-checkpoints",
    )
    parser.add_argument(
        "--output-csv",
        default="results/v11_screen_summary.csv",
    )
    args = parser.parse_args()

    root = Path(args.checkpoint_root)
    rows = []
    missing = []
    for run in SCREEN_RUNS:
        checkpoint_path = root / run / "best.pth"
        if checkpoint_path.exists():
            rows.append(load_row(checkpoint_path, run))
        else:
            missing.append(str(checkpoint_path))
    if not rows:
        raise FileNotFoundError(
            "No V1.1 screen checkpoints were found. Expected:\n"
            + "\n".join(missing)
        )

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    headers = (
        "run",
        "step",
        "psnr",
        "ssim",
        "lpips",
        "dists",
        "lowlight_psnr",
        "triple_psnr",
    )
    print(" | ".join(headers))
    print(" | ".join("---" for _ in headers))
    for row in rows:
        print(" | ".join(format_value(row[name]) for name in headers))
    if missing:
        print(f"Skipped {len(missing)} missing checkpoints.")
    print(f"CSV: {output_path.resolve()}")


if __name__ == "__main__":
    main()
