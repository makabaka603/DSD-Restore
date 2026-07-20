import argparse
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a short, representative training benchmark config")
    parser.add_argument("--config", default="configs/train_v1_minimal.yaml")
    parser.add_argument("--output", default="/tmp/dsd_benchmark.yaml")
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--disable-compile", action="store_true")
    args = parser.parse_args()
    if args.iters <= 0:
        raise ValueError("--iters must be greater than zero")

    with open(args.config, encoding="utf-8") as config_file:
        cfg = yaml.safe_load(config_file)
    cfg["optimization"]["max_iters"] = args.iters
    runtime = cfg["runtime"]
    runtime["val_every_iters"] = args.iters
    runtime["best_window_iters"] = args.iters
    runtime["selection_val_every_iters"] = args.iters
    runtime["selection_val_max_samples_per_source"] = 2
    runtime["save_every_iters"] = args.iters
    runtime["early_stop_patience_iters"] = 0
    runtime["checkpoint_dir"] = "/root/autodl-tmp/DSD-Restor-checkpoints/benchmark"
    runtime["tensorboard"]["run_name"] = "benchmark"
    if args.disable_compile:
        runtime["compile"]["enabled"] = False

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as output_file:
        yaml.safe_dump(cfg, output_file, sort_keys=False, allow_unicode=True)
    print(f"Wrote {output}")
    print(f"Run: /usr/bin/time -p python train.py --config {output}")


if __name__ == "__main__":
    main()
