import argparse
import bisect
import sys
from collections import Counter
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets import build_balanced_sampler, build_source_dataset
from utils.config import load_config


def check_pair_sizes(sources, max_pairs: int) -> None:
    for source in sources:
        pairs = source.pairs if max_pairs <= 0 else source.pairs[:max_pairs]
        mismatches = []
        for input_path, gt_path in pairs:
            with Image.open(input_path) as image, Image.open(gt_path) as target:
                if image.size != target.size:
                    mismatches.append((input_path, image.size, gt_path, target.size))
                    if len(mismatches) == 5:
                        break
        if mismatches:
            details = "\n".join(
                f"  {input_path} {input_size} != {gt_path} {gt_size}"
                for input_path, input_size, gt_path, gt_size in mismatches
            )
            raise ValueError(f"{source.name} contains mismatched pairs:\n{details}")
        print(f"  size check passed: {source.name} ({len(pairs)} pairs)")


def check_split_overlap(train_sources, val_sources) -> None:
    train_paths = {
        path.absolute()
        for source in train_sources
        for pair in source.pairs
        for path in pair
    }
    val_paths = {
        path.absolute()
        for source in val_sources
        for pair in source.pairs
        for path in pair
    }
    overlap = train_paths & val_paths
    if overlap:
        preview = "\n".join(f"  {path}" for path in sorted(overlap)[:10])
        raise ValueError(f"Training/validation leakage detected ({len(overlap)} files):\n{preview}")
    print("  split overlap check passed")


def report_sampling(train_sources, source_configs, seed: int, draws: int) -> None:
    sampler = build_balanced_sampler(train_sources, source_configs, num_samples=draws, seed=seed)
    boundaries, running = [], 0
    for source in train_sources:
        running += len(source)
        boundaries.append(running)
    counts = Counter()
    for index in sampler:
        source_index = bisect.bisect_right(boundaries, index)
        counts[train_sources[source_index].name] += 1
    print(f"  empirical sampling distribution ({draws} draws):")
    for source, config in zip(train_sources, source_configs):
        actual = counts[source.name] / draws
        print(
            f"    {source.name:<24} target={float(config['probability']):6.2%} "
            f"actual={actual:6.2%}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit all configured restoration training sources")
    parser.add_argument("--config", default="configs/train_v1_stage1.yaml")
    parser.add_argument(
        "--check-sizes",
        action="store_true",
        help="open image headers and verify every configured input/GT size",
    )
    parser.add_argument(
        "--max-pairs-per-source",
        type=int,
        default=0,
        help="limit size checks per source; 0 checks every pair",
    )
    parser.add_argument("--draws", type=int, default=100000)
    args = parser.parse_args()

    cfg = load_config(args.config)
    data = cfg["data"]
    if not data.get("train_sources") or not data.get("val_sources"):
        raise ValueError("The config must define data.train_sources and data.val_sources")
    probability_sum = sum(float(source["probability"]) for source in data["train_sources"])
    if abs(probability_sum - 1.0) > 1e-6:
        raise ValueError(f"Training source probabilities must sum to 1.0, got {probability_sum}")

    seed = cfg["experiment"].get("seed", 42)
    train_set, train_sources = build_source_dataset(
        data["train_sources"], data.get("crop_size"), training=True, seed=seed
    )
    val_set, val_sources = build_source_dataset(
        data["val_sources"], crop_size=None, training=False, seed=seed
    )
    print(f"train images: {len(train_set)}")
    print(f"validation images: {len(val_set)}")
    for source in train_sources:
        print(f"  train/{source.name}: {len(source)}")
    for source in val_sources:
        print(f"  val/{source.name}: {len(source)}")

    check_split_overlap(train_sources, val_sources)
    report_sampling(train_sources, data["train_sources"], seed, args.draws)
    if args.check_sizes:
        check_pair_sizes(train_sources + val_sources, args.max_pairs_per_source)
    else:
        print("  size check skipped (pass --check-sizes before formal training)")
    print("Training data audit passed")


if __name__ == "__main__":
    main()
