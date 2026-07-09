import json
import random
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
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
            self.metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))

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
        labels = self._labels(input_path.name)
        return {
            "input": image,
            "gt": target,
            "dense_label": labels["dense"],
            "sparse_label": labels["sparse"],
            "name": input_path.name,
        }

    def _paired_transform(self, image: Image.Image, target: Image.Image) -> tuple[torch.Tensor, torch.Tensor]:
        if self.training and self.crop_size:
            w, h = image.size
            crop = min(self.crop_size, w, h)
            left = random.randint(0, max(0, w - crop))
            top = random.randint(0, max(0, h - crop))
            image = TF.crop(image, top, left, crop, crop)
            target = TF.crop(target, top, left, crop, crop)
            if random.random() < 0.5:
                image = TF.hflip(image)
                target = TF.hflip(target)
            if random.random() < 0.5:
                image = TF.vflip(image)
                target = TF.vflip(target)
        return TF.to_tensor(image), TF.to_tensor(target)

    def _labels(self, name: str) -> dict[str, torch.Tensor]:
        meta = self.metadata.get(name, {})
        dense = torch.tensor([float(meta.get(key, 0.0)) for key in self.dense_keys], dtype=torch.float32)
        sparse = torch.tensor([float(meta.get(key, 0.0)) for key in self.sparse_keys], dtype=torch.float32)
        return {"dense": dense.clamp(0, 1), "sparse": sparse.clamp(0, 1)}
