import json
import random
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF


class PairedRestorationDataset(Dataset):
    dense_keys = ("dust", "sand", "haze", "lowlight", "colorcast")
    sparse_keys = ("rain", "raindrop", "snow", "occlusion")
    image_suffixes = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}

    def __init__(
        self,
        input_dir: str,
        gt_dir: str,
        metadata_path: str | None = None,
        crop_size: int | None = None,
        training: bool = True,
    ):
        self.input_dir = Path(input_dir)
        self.gt_dir = Path(gt_dir)
        self.crop_size = crop_size
        self.training = training
        if not self.input_dir.exists() or not self.gt_dir.exists():
            raise FileNotFoundError(f"Dataset folders not found: {self.input_dir}, {self.gt_dir}")
        self.files = sorted([p for p in self.input_dir.iterdir() if p.suffix.lower() in self.image_suffixes])
        if not self.files:
            raise FileNotFoundError(f"No images found in {self.input_dir}")
        self.metadata = {}
        if metadata_path and Path(metadata_path).exists():
            metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
            # Generated benchmarks store dataset-level information alongside
            # per-file records under "samples". Legacy metadata is already a
            # direct filename-to-label mapping.
            self.metadata = metadata.get("samples", metadata)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        input_path = self.files[index]
        gt_path = self.gt_dir / input_path.name
        if not gt_path.exists():
            raise FileNotFoundError(f"Missing GT pair for {input_path.name}: {gt_path}")
        image = Image.open(input_path).convert("RGB")
        target = Image.open(gt_path).convert("RGB")
        image, target = self._paired_transform(image, target)
        meta = self.metadata.get(input_path.name, {})
        labels = self._labels(meta)
        tasks = meta.get("tasks")
        if not tasks:
            strengths = meta.get("strengths", meta)
            tasks = [
                key
                for key in (*self.dense_keys, *self.sparse_keys)
                if float(strengths.get(key, 0.0)) > 0
            ]
        return {
            "input": image,
            "gt": target,
            "dense_label": labels["dense"],
            "sparse_label": labels["sparse"],
            "name": input_path.name,
            "source": meta.get("category", "paired"),
            "task": "+".join(tasks) if tasks else "mixed",
        }

    def _paired_transform(self, image: Image.Image, target: Image.Image) -> tuple[torch.Tensor, torch.Tensor]:
        if image.size != target.size:
            raise ValueError(
                f"Input/GT size mismatch before paired transform: input={image.size}, gt={target.size}"
            )
        if self.crop_size:
            w, h = image.size
            crop = self.crop_size
            if min(w, h) < crop:
                scale = crop / min(w, h)
                resized_w = max(crop, round(w * scale))
                resized_h = max(crop, round(h * scale))
                size = [resized_h, resized_w]
                image = TF.resize(image, size, interpolation=InterpolationMode.BICUBIC, antialias=True)
                target = TF.resize(target, size, interpolation=InterpolationMode.BICUBIC, antialias=True)
                w, h = image.size
            if self.training:
                left = random.randint(0, max(0, w - crop))
                top = random.randint(0, max(0, h - crop))
            else:
                left = max(0, (w - crop) // 2)
                top = max(0, (h - crop) // 2)
            image = TF.crop(image, top, left, crop, crop)
            target = TF.crop(target, top, left, crop, crop)
            if self.training:
                if random.random() < 0.5:
                    image = TF.hflip(image)
                    target = TF.hflip(target)
                if random.random() < 0.5:
                    image = TF.vflip(image)
                    target = TF.vflip(target)
        return TF.to_tensor(image), TF.to_tensor(target)

    def _labels(self, meta: dict) -> dict[str, torch.Tensor]:
        strengths = meta.get("strengths", meta)
        dense = torch.tensor(
            [float(strengths.get(key, 0.0)) for key in self.dense_keys], dtype=torch.float32
        )
        sparse = torch.tensor(
            [float(strengths.get(key, 0.0)) for key in self.sparse_keys], dtype=torch.float32
        )
        return {"dense": dense.clamp(0, 1), "sparse": sparse.clamp(0, 1)}
