import argparse
import sys
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from torchvision.transforms import functional as TF

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets import synthesize_composite_degradation


def fit_preview(image: Image.Image, size: int = 384) -> Image.Image:
    image = image.copy()
    image.thumbnail((size, size), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (size, size), "white")
    left = (size - image.width) // 2
    top = (size - image.height) // 2
    canvas.paste(image, (left, top))
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(description="Preview DIV2K synthetic dense degradations")
    parser.add_argument("--input", default=None, help="clean image; defaults to first DIV2K train image")
    parser.add_argument("--output", default="results/synthetic_preview.jpg")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.input:
        input_path = Path(args.input)
    else:
        candidates = sorted(Path("datasets/DIV2K_train_HR").glob("*"))
        if not candidates:
            raise FileNotFoundError("No DIV2K training images found")
        input_path = candidates[0]
    clean_image = Image.open(input_path).convert("RGB")
    # Synthesize at the displayed scale so sparse rain/snow elements are not
    # erased by thumbnailing a much larger DIV2K image afterward.
    clean_preview = fit_preview(clean_image)
    clean = TF.to_tensor(clean_preview)

    panels = [("clean", clean_preview)]
    preview_tasks = (
        ("dust",),
        ("sand",),
        ("haze", "lowlight"),
        ("haze", "rain"),
        ("lowlight", "snow"),
        ("dust", "lowlight", "rain"),
    )
    for offset, tasks in enumerate(preview_tasks):
        generator = torch.Generator().manual_seed(args.seed + offset)
        degraded, _ = synthesize_composite_degradation(clean, tasks, generator)
        panels.append(("+".join(tasks), fit_preview(TF.to_pil_image(degraded))))

    label_height = 36
    montage = Image.new("RGB", (384 * len(panels), 384 + label_height), "white")
    draw = ImageDraw.Draw(montage)
    for index, (label, panel) in enumerate(panels):
        x = index * 384
        montage.paste(panel, (x, label_height))
        draw.text((x + 12, 10), label, fill="black")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    montage.save(output_path, quality=95)
    print(f"Saved synthetic degradation preview to {output_path}")


if __name__ == "__main__":
    main()
