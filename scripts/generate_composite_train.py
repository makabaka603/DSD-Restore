import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm
from torchvision.transforms import functional as TF

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets import synthesize_composite_degradation


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}

# These weights represent the synthetic 60% of the full V1 sampler. The other
# 40% comes from RESIDE/LOLv2 (20%) and Rain13K/Snow100K (20%).
OFFLINE_PLAN = {
    "single_dense": {
        "weight": 10,
        "mixtures": [("dust",), ("sand",), ("colorcast",)],
    },
    "dense_dense": {
        "weight": 25,
        "mixtures": [
            ("dust", "haze"),
            ("dust", "lowlight"),
            ("sand", "haze"),
            ("haze", "lowlight"),
        ],
    },
    "dense_sparse": {
        "weight": 20,
        "mixtures": [
            ("dust", "rain"),
            ("haze", "rain"),
            ("lowlight", "rain"),
            ("haze", "snow"),
            ("lowlight", "snow"),
        ],
    },
    "triple": {
        "weight": 5,
        "mixtures": [
            ("dust", "haze", "lowlight"),
            ("dust", "lowlight", "rain"),
            ("haze", "lowlight", "rain"),
            ("haze", "lowlight", "snow"),
        ],
    },
}


def deterministic_crop(
    image: Image.Image,
    size: int,
    generator: torch.Generator,
) -> Image.Image:
    image = image.convert("RGB")
    width, height = image.size
    if min(width, height) < size:
        scale = size / min(width, height)
        image = image.resize(
            (max(size, round(width * scale)), max(size, round(height * scale))),
            Image.Resampling.LANCZOS,
        )
        width, height = image.size
    left = int(torch.randint(0, width - size + 1, (), generator=generator).item())
    top = int(torch.randint(0, height - size + 1, (), generator=generator).item())
    return image.crop((left, top, left + size, top + size))


def require_empty_output(output_dir: Path) -> tuple[Path, Path]:
    input_dir = output_dir / "input"
    gt_dir = output_dir / "gt"
    existing = []
    for folder in (input_dir, gt_dir):
        if folder.exists():
            existing.extend(path for path in folder.iterdir() if path.is_file())
    if existing or (output_dir / "metadata.json").exists():
        raise FileExistsError(
            f"{output_dir} already contains generated data. Use a new --output path; "
            "the generator never overwrites a training or validation set."
        )
    input_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)
    return input_dir, gt_dir


def allocate_counts(total_samples: int) -> dict[str, int]:
    if total_samples <= 0:
        return {group: 0 for group in OFFLINE_PLAN}
    total_weight = sum(int(spec["weight"]) for spec in OFFLINE_PLAN.values())
    counts = {
        group: total_samples * int(spec["weight"]) // total_weight
        for group, spec in OFFLINE_PLAN.items()
    }
    remainder = total_samples - sum(counts.values())
    groups = list(OFFLINE_PLAN)
    for index in range(remainder):
        counts[groups[index % len(groups)]] += 1
    return counts


def build_specs(
    clean_files: list[Path],
    output_dir: Path,
    total_samples: int,
    size: int,
    seed: int,
) -> tuple[list[tuple], dict[str, int]]:
    counts = allocate_counts(total_samples)
    specs = []
    global_index = 0
    for group_index, (group, group_spec) in enumerate(OFFLINE_PLAN.items()):
        mixtures = group_spec["mixtures"]
        for local_index in range(counts[group]):
            tasks = mixtures[local_index % len(mixtures)]
            clean_path = clean_files[(global_index * 37 + group_index) % len(clean_files)]
            sample_seed = seed + group_index * 1_000_003 + local_index * 10_007
            filename = f"{global_index:06d}_{group}.png"
            specs.append(
                (
                    filename,
                    group,
                    tasks,
                    str(clean_path),
                    sample_seed,
                    size,
                    str(output_dir / "input"),
                    str(output_dir / "gt"),
                )
            )
            global_index += 1
    if global_index != total_samples:
        raise RuntimeError(f"Plan produced {global_index} samples, expected {total_samples}")
    return specs, counts


def initialize_worker() -> None:
    # Multiple PyTorch thread pools inside multiple processes cause severe CPU
    # oversubscription. Each process only needs one thread for these small crops.
    torch.set_num_threads(1)


def generate_one(spec: tuple) -> tuple[str, dict]:
    filename, group, tasks, clean_path_text, sample_seed, size, input_text, gt_text = spec
    clean_path = Path(clean_path_text)
    generator = torch.Generator().manual_seed(sample_seed)
    with Image.open(clean_path) as opened:
        target_image = deterministic_crop(opened, size, generator)
    target = TF.to_tensor(target_image)
    degraded, strengths = synthesize_composite_degradation(target, tasks, generator)
    target_image.save(Path(gt_text) / filename, compress_level=3)
    TF.to_pil_image(degraded).save(Path(input_text) / filename, compress_level=3)
    return filename, {
        "group": group,
        "category": "+".join(tasks),
        "tasks": list(tasks),
        "strengths": strengths,
        "source": clean_path.name,
        "seed": sample_seed,
    }


def generate_dataset(
    name: str,
    clean_files: list[Path],
    output_dir: Path,
    total_samples: int,
    size: int,
    seed: int,
    workers: int,
) -> None:
    if total_samples <= 0:
        print(f"Skipping {name}: sample count is zero")
        return
    require_empty_output(output_dir)
    specs, counts = build_specs(clean_files, output_dir, total_samples, size, seed)
    metadata = {
        "name": name,
        "seed": seed,
        "crop_size": size,
        "clean_sources": [path.name for path in clean_files],
        "group_counts": counts,
        "samples": {},
    }

    if workers <= 1:
        results = map(generate_one, specs)
        for filename, record in tqdm(results, total=len(specs), desc=name):
            metadata["samples"][filename] = record
    else:
        with ProcessPoolExecutor(
            max_workers=workers, initializer=initialize_worker
        ) as executor:
            results = executor.map(generate_one, specs, chunksize=8)
            for filename, record in tqdm(results, total=len(specs), desc=name):
                metadata["samples"][filename] = record

    metadata_tmp = output_dir / "metadata.json.tmp"
    metadata_tmp.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    metadata_tmp.replace(output_dir / "metadata.json")
    print(f"Generated {len(metadata['samples'])} paired samples in {output_dir}")
    print(f"Group counts: {counts}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Materialize offline synthetic composite training and validation sets"
    )
    parser.add_argument("--clean-dir", default="datasets/DIV2K_train_HR")
    parser.add_argument("--train-output", default="datasets/SyntheticCompositeTrain")
    parser.add_argument("--val-output", default="datasets/SyntheticCompositeVal")
    parser.add_argument("--train-samples", type=int, default=60000)
    parser.add_argument("--val-samples", type=int, default=1200)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--val-clean-count", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument(
        "--workers", type=int, default=min(8, max(1, os.cpu_count() or 1))
    )
    args = parser.parse_args()

    if args.size <= 0:
        raise ValueError("--size must be greater than zero")
    if args.train_samples < 0 or args.val_samples < 0:
        raise ValueError("sample counts cannot be negative")
    if args.workers <= 0:
        raise ValueError("--workers must be greater than zero")
    clean_dir = Path(args.clean_dir)
    if not clean_dir.exists():
        raise FileNotFoundError(f"Clean image folder not found: {clean_dir}")
    clean_files = sorted(
        path for path in clean_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES
    )
    if args.val_clean_count <= 0 or args.val_clean_count >= len(clean_files):
        raise ValueError(
            f"--val-clean-count must be between 1 and {len(clean_files) - 1}"
        )
    train_clean = clean_files[: -args.val_clean_count]
    val_clean = clean_files[-args.val_clean_count :]
    if set(train_clean) & set(val_clean):
        raise RuntimeError("Training and validation clean-image splits overlap")

    generate_dataset(
        "SyntheticCompositeTrain",
        train_clean,
        Path(args.train_output),
        args.train_samples,
        args.size,
        args.seed,
        args.workers,
    )
    generate_dataset(
        "SyntheticCompositeVal",
        val_clean,
        Path(args.val_output),
        args.val_samples,
        args.size,
        args.seed + 99_999_937,
        args.workers,
    )


if __name__ == "__main__":
    main()
