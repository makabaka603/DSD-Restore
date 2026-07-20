import hashlib
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import ConcatDataset, Dataset, WeightedRandomSampler
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
DENSE_KEYS = ("dust", "sand", "haze", "lowlight", "colorcast")
SPARSE_KEYS = ("rain", "raindrop", "snow", "occlusion")


def _stable_seed(*parts: str, base_seed: int = 0) -> int:
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).digest()
    return (int.from_bytes(digest[:8], "little") + base_seed) % (2**63 - 1)


def _label_vector(tasks: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
    unknown = sorted(set(tasks) - set(DENSE_KEYS) - set(SPARSE_KEYS))
    if unknown:
        raise ValueError(f"Unknown degradation labels: {unknown}")
    dense = torch.tensor([float(key in tasks) for key in DENSE_KEYS], dtype=torch.float32)
    sparse = torch.tensor([float(key in tasks) for key in SPARSE_KEYS], dtype=torch.float32)
    return dense, sparse


def _uniform(generator: torch.Generator, low: float, high: float) -> float:
    return low + (high - low) * float(torch.rand((), generator=generator))


def synthesize_degradation(
    clean: torch.Tensor,
    degradation: str,
    generator: torch.Generator,
) -> torch.Tensor:
    """Generate a paired dense degradation from a clean CHW tensor in [0, 1]."""
    if degradation == "colorcast":
        gains = torch.tensor(
            [_uniform(generator, 0.65, 1.35) for _ in range(3)],
            dtype=clean.dtype,
        ).view(3, 1, 1)
        if float((gains - 1.0).abs().max()) < 0.12:
            gains[0] = 1.25
            gains[2] = 0.78
        gamma = _uniform(generator, 0.85, 1.2)
        return (clean.clamp_min(1e-6).pow(gamma) * gains).clamp(0, 1)

    if degradation not in {"dust", "sand"}:
        raise ValueError(f"Unsupported synthetic degradation: {degradation}")

    _, height, width = clean.shape
    grid_h = max(2, min(8, height // 64))
    grid_w = max(2, min(8, width // 64))
    field = torch.rand((1, 1, grid_h, grid_w), generator=generator, dtype=clean.dtype)
    field = F.interpolate(field, size=(height, width), mode="bicubic", align_corners=False)[0]
    field = (field - field.amin()) / (field.amax() - field.amin() + 1e-6)

    if degradation == "dust":
        transmission = _uniform(generator, 0.48, 0.72) + field * _uniform(generator, 0.08, 0.22)
        atmosphere = torch.tensor(
            [
                _uniform(generator, 0.78, 0.96),
                _uniform(generator, 0.60, 0.82),
                _uniform(generator, 0.38, 0.62),
            ],
            dtype=clean.dtype,
        ).view(3, 1, 1)
        noise_std = _uniform(generator, 0.003, 0.012)
    else:
        transmission = _uniform(generator, 0.38, 0.62) + field * _uniform(generator, 0.06, 0.18)
        atmosphere = torch.tensor(
            [
                _uniform(generator, 0.86, 1.0),
                _uniform(generator, 0.68, 0.88),
                _uniform(generator, 0.30, 0.52),
            ],
            dtype=clean.dtype,
        ).view(3, 1, 1)
        noise_std = _uniform(generator, 0.008, 0.025)

    transmission = transmission.clamp(0.25, 0.92)
    degraded = clean * transmission + atmosphere * (1.0 - transmission)
    noise = torch.randn(clean.shape, generator=generator, dtype=clean.dtype) * noise_std
    return (degraded + noise).clamp(0, 1)


class RestorationSourceDataset(Dataset):
    def __init__(
        self,
        config: dict,
        crop_size: int | None,
        training: bool,
        seed: int,
    ):
        self.name = config["name"]
        self.tasks = config.get("tasks", [config.get("task")])
        self.tasks = [task for task in self.tasks if task]
        self.dense_label, self.sparse_label = _label_vector(self.tasks)
        self.crop_size = crop_size
        self.training = training
        self.seed = seed
        self.synthetic = config.get("synthetic")

        if self.synthetic:
            clean_dir = Path(config["clean_dir"])
            if not clean_dir.exists():
                raise FileNotFoundError(f"Clean image folder not found for {self.name}: {clean_dir}")
            clean_files = sorted(p for p in clean_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
            self.pairs = [(path, path) for path in clean_files]
        else:
            input_dir = Path(config["input_dir"])
            gt_dir = Path(config["gt_dir"])
            if not input_dir.exists() or not gt_dir.exists():
                raise FileNotFoundError(
                    f"Dataset folders not found for {self.name}: {input_dir}, {gt_dir}"
                )
            input_files = sorted(p for p in input_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
            gt_names = {
                path.name for path in gt_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES
            }
            strip_prefix = config.get("gt_strip_input_prefix", "")
            self.pairs = []
            for path in input_files:
                gt_name = path.name
                if strip_prefix and gt_name.startswith(strip_prefix):
                    gt_name = gt_name[len(strip_prefix):]
                self.pairs.append((path, gt_dir / gt_name))
            missing = [str(gt) for _, gt in self.pairs if gt.name not in gt_names]
            if missing:
                preview = "\n".join(missing[:5])
                raise FileNotFoundError(
                    f"{self.name} has {len(missing)} inputs without same-name GT files. Examples:\n{preview}"
                )

        max_samples = config.get("max_samples")
        if max_samples and len(self.pairs) > max_samples:
            if max_samples == 1:
                selected = [0]
            else:
                selected = [round(i * (len(self.pairs) - 1) / (max_samples - 1)) for i in range(max_samples)]
            self.pairs = [self.pairs[index] for index in selected]
        if not self.pairs:
            raise FileNotFoundError(f"No images found for source {self.name}")

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        input_path, gt_path = self.pairs[index]
        with Image.open(input_path) as opened_image:
            image = opened_image.convert("RGB")
        if input_path == gt_path:
            target = image.copy()
        else:
            with Image.open(gt_path) as opened_target:
                target = opened_target.convert("RGB")
        image, target = self._paired_transform(image, target)

        if self.synthetic:
            if self.training:
                # DataLoader seeds each worker's global torch RNG. Drawing the
                # per-sample seed from it keeps augmentation random and reproducible.
                sample_seed = int(torch.randint(0, 2**63 - 1, ()).item())
                generator = torch.Generator().manual_seed(sample_seed)
            else:
                generator = torch.Generator().manual_seed(
                    _stable_seed(self.name, input_path.name, base_seed=self.seed)
                )
            image = synthesize_degradation(target, self.synthetic, generator)

        return {
            "input": image,
            "gt": target,
            "dense_label": self.dense_label.clone(),
            "sparse_label": self.sparse_label.clone(),
            "name": input_path.name,
            "source": self.name,
            "task": "+".join(self.tasks),
        }

    def _paired_transform(self, image: Image.Image, target: Image.Image) -> tuple[torch.Tensor, torch.Tensor]:
        if image.size != target.size:
            raise ValueError(
                f"{self.name}: input/GT size mismatch: input={image.size}, gt={target.size}"
            )
        if self.training and self.crop_size:
            width, height = image.size
            crop = self.crop_size
            if min(width, height) < crop:
                scale = crop / min(width, height)
                size = [max(crop, round(height * scale)), max(crop, round(width * scale))]
                image = TF.resize(image, size, interpolation=InterpolationMode.BICUBIC, antialias=True)
                target = TF.resize(target, size, interpolation=InterpolationMode.BICUBIC, antialias=True)
                width, height = image.size
            left = random.randint(0, max(0, width - crop))
            top = random.randint(0, max(0, height - crop))
            image = TF.crop(image, top, left, crop, crop)
            target = TF.crop(target, top, left, crop, crop)
            if random.random() < 0.5:
                image, target = TF.hflip(image), TF.hflip(target)
            if random.random() < 0.5:
                image, target = TF.vflip(image), TF.vflip(target)
        return TF.to_tensor(image), TF.to_tensor(target)


def build_source_dataset(
    source_configs: list[dict],
    crop_size: int | None,
    training: bool,
    seed: int,
) -> tuple[ConcatDataset, list[RestorationSourceDataset]]:
    sources = [
        RestorationSourceDataset(config, crop_size=crop_size, training=training, seed=seed)
        for config in source_configs
    ]
    return ConcatDataset(sources), sources


def build_balanced_sampler(
    sources: list[RestorationSourceDataset],
    source_configs: list[dict],
    num_samples: int,
    seed: int,
) -> WeightedRandomSampler:
    probabilities = [float(config["probability"]) for config in source_configs]
    total = sum(probabilities)
    if total <= 0:
        raise ValueError("Training source probabilities must sum to a positive value")
    probabilities = [probability / total for probability in probabilities]
    weights = []
    for source, probability in zip(sources, probabilities):
        weights.extend([probability / len(source)] * len(source))
    generator = torch.Generator().manual_seed(seed)
    return WeightedRandomSampler(
        torch.as_tensor(weights, dtype=torch.double),
        num_samples=num_samples,
        replacement=True,
        generator=generator,
    )
