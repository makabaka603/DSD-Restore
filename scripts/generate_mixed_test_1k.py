import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms import functional as TF

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets import synthesize_composite_degradation


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
MIXED_TEST_PLAN = {
    "dust_haze": (150, [("dust", "haze")]),
    "dust_lowlight": (150, [("dust", "lowlight")]),
    "haze_lowlight": (150, [("haze", "lowlight")]),
    "dust_rain": (100, [("dust", "rain")]),
    "haze_rain": (100, [("haze", "rain")]),
    "lowlight_rain": (100, [("lowlight", "rain")]),
    "haze_snow": (75, [("haze", "snow")]),
    "lowlight_snow": (75, [("lowlight", "snow")]),
    "triple": (
        100,
        [
            ("dust", "haze", "lowlight"),
            ("dust", "lowlight", "rain"),
            ("haze", "lowlight", "rain"),
            ("haze", "lowlight", "snow"),
        ],
    ),
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
            f"{output_dir} already contains a generated dataset. "
            "Use a new --output directory so an existing benchmark is never overwritten."
        )
    input_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)
    return input_dir, gt_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate the fixed Synthetic-Mixed-Test-1K benchmark"
    )
    parser.add_argument("--clean-dir", default="datasets/DIV2K_valid_HR")
    parser.add_argument("--output", default="datasets/Synthetic-Mixed-Test-1K")
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260722)
    args = parser.parse_args()

    if args.size <= 0:
        raise ValueError("--size must be greater than zero")
    clean_dir = Path(args.clean_dir)
    clean_files = sorted(
        path for path in clean_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not clean_files:
        raise FileNotFoundError(f"No clean images found in {clean_dir}")

    output_dir = Path(args.output)
    input_dir, gt_dir = require_empty_output(output_dir)
    metadata = {
        "name": "Synthetic-Mixed-Test-1K",
        "seed": args.seed,
        "crop_size": args.size,
        "clean_source": str(clean_dir),
        "samples": {},
    }

    global_index = 0
    for category_index, (category, (count, mixture_pool)) in enumerate(MIXED_TEST_PLAN.items()):
        for local_index in range(count):
            # The coprime stride spreads repeated uses across the clean set.
            clean_path = clean_files[(global_index * 37 + category_index) % len(clean_files)]
            sample_seed = args.seed + category_index * 1_000_003 + local_index * 10_007
            generator = torch.Generator().manual_seed(sample_seed)
            with Image.open(clean_path) as opened:
                target_image = deterministic_crop(opened, args.size, generator)
            target = TF.to_tensor(target_image)
            tasks = mixture_pool[local_index % len(mixture_pool)]
            degraded, strengths = synthesize_composite_degradation(
                target, tasks, generator
            )

            filename = f"{global_index:04d}_{category}.png"
            target_image.save(gt_dir / filename)
            TF.to_pil_image(degraded).save(input_dir / filename)
            metadata["samples"][filename] = {
                "category": category,
                "tasks": list(tasks),
                "strengths": strengths,
                "source": clean_path.name,
                "seed": sample_seed,
            }
            global_index += 1

    if global_index != 1000:
        raise RuntimeError(f"Mixed-Test plan produced {global_index} samples, expected 1000")
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Generated {global_index} paired samples in {output_dir}")


if __name__ == "__main__":
    main()
