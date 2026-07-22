import hashlib
import json
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


def _strength_vector(strengths: dict[str, float]) -> tuple[torch.Tensor, torch.Tensor]:
    dense = torch.tensor(
        [float(strengths.get(key, 0.0)) for key in DENSE_KEYS], dtype=torch.float32
    )
    sparse = torch.tensor(
        [float(strengths.get(key, 0.0)) for key in SPARSE_KEYS], dtype=torch.float32
    )
    return dense.clamp(0, 1), sparse.clamp(0, 1)


def _uniform(generator: torch.Generator, low: float, high: float) -> float:
    return low + (high - low) * float(torch.rand((), generator=generator))


def _smooth_random_field(
    image: torch.Tensor,
    generator: torch.Generator,
    min_grid: int = 2,
    max_grid: int = 8,
) -> torch.Tensor:
    _, height, width = image.shape
    grid_h = max(2, min(8, height // 64))
    grid_w = max(2, min(8, width // 64))
    grid_h = max(min_grid, min(max_grid, grid_h))
    grid_w = max(min_grid, min(max_grid, grid_w))
    field = torch.rand((1, 1, grid_h, grid_w), generator=generator, dtype=image.dtype)
    field = F.interpolate(field, size=(height, width), mode="bicubic", align_corners=False)[0]
    return (field - field.amin()) / (field.amax() - field.amin() + 1e-6)


def _apply_dust_or_sand(
    image: torch.Tensor,
    degradation: str,
    generator: torch.Generator,
) -> tuple[torch.Tensor, float]:
    field = _smooth_random_field(image, generator)
    strength = _uniform(generator, 0.35, 0.85)

    if degradation == "dust":
        base_transmission = 0.86 - 0.50 * strength
        transmission = base_transmission + field * _uniform(generator, 0.08, 0.22)
        atmosphere = torch.tensor(
            [
                _uniform(generator, 0.78, 0.96),
                _uniform(generator, 0.60, 0.82),
                _uniform(generator, 0.38, 0.62),
            ],
            dtype=image.dtype,
        ).view(3, 1, 1)
        noise_std = _uniform(generator, 0.003, 0.012) * strength
    else:
        base_transmission = 0.78 - 0.48 * strength
        transmission = base_transmission + field * _uniform(generator, 0.06, 0.18)
        atmosphere = torch.tensor(
            [
                _uniform(generator, 0.86, 1.0),
                _uniform(generator, 0.68, 0.88),
                _uniform(generator, 0.30, 0.52),
            ],
            dtype=image.dtype,
        ).view(3, 1, 1)
        noise_std = _uniform(generator, 0.008, 0.025) * strength

    transmission = transmission.clamp(0.25, 0.92)
    degraded = image * transmission + atmosphere * (1.0 - transmission)
    noise = torch.randn(image.shape, generator=generator, dtype=image.dtype) * noise_std
    return (degraded + noise).clamp(0, 1), strength


def _apply_haze(
    image: torch.Tensor,
    generator: torch.Generator,
) -> tuple[torch.Tensor, float]:
    strength = _uniform(generator, 0.20, 0.80)
    field = _smooth_random_field(image, generator)
    transmission = (1.0 - 0.65 * strength) + (field - 0.5) * 0.12
    transmission = transmission.clamp(0.25, 0.95)
    atmospheric_light = torch.tensor(
        [_uniform(generator, 0.82, 1.0) for _ in range(3)], dtype=image.dtype
    ).view(3, 1, 1)
    degraded = image * transmission + atmospheric_light * (1.0 - transmission)
    return degraded.clamp(0, 1), strength


def _apply_lowlight(
    image: torch.Tensor,
    generator: torch.Generator,
) -> tuple[torch.Tensor, float]:
    strength = _uniform(generator, 0.25, 0.85)
    illumination = _smooth_random_field(image, generator, min_grid=2, max_grid=5)
    illumination = (0.30 + 0.55 * illumination) * (1.0 - 0.45 * strength)
    gamma = 1.0 + 1.6 * strength
    degraded = image.clamp_min(1e-6).pow(gamma) * illumination
    noise_std = 0.003 + 0.025 * strength
    noise = torch.randn(image.shape, generator=generator, dtype=image.dtype) * noise_std
    return (degraded + noise).clamp(0, 1), strength


def _apply_colorcast(
    image: torch.Tensor,
    generator: torch.Generator,
) -> tuple[torch.Tensor, float]:
    strength = _uniform(generator, 0.20, 0.75)
    warm = float(torch.rand((), generator=generator)) < 0.65
    if warm:
        target_gains = torch.tensor([1.30, 1.08, 0.72], dtype=image.dtype)
    else:
        target_gains = torch.tensor([0.78, 1.02, 1.25], dtype=image.dtype)
    gains = 1.0 + (target_gains - 1.0) * strength
    return (image * gains.view(3, 1, 1)).clamp(0, 1), strength


def _apply_rain(
    image: torch.Tensor,
    generator: torch.Generator,
) -> tuple[torch.Tensor, float]:
    _, height, width = image.shape
    strength = _uniform(generator, 0.20, 0.80)
    density = 0.002 + 0.008 * strength
    impulses = (
        torch.rand((1, 1, height, width), generator=generator, dtype=image.dtype) < density
    ).to(image.dtype)
    length = 9 + 2 * int(round(7 * strength))
    kernel = torch.zeros((1, 1, length, length), dtype=image.dtype)
    direction = int(torch.randint(0, 3, (), generator=generator).item())
    if direction == 0:
        kernel[0, 0, :, length // 2] = 1.0
    elif direction == 1:
        kernel[0, 0].diagonal().fill_(1.0)
    else:
        kernel[0, 0] = torch.fliplr(torch.eye(length, dtype=image.dtype))
    kernel /= kernel.sum().clamp_min(1.0)
    streaks = F.conv2d(impulses, kernel, padding=length // 2)[0]
    streaks = streaks / streaks.amax().clamp_min(1e-6)
    alpha = streaks * (0.10 + 0.30 * strength)
    rain_color = torch.tensor([0.82, 0.88, 0.95], dtype=image.dtype).view(3, 1, 1)
    degraded = image * (1.0 - alpha) + rain_color * alpha
    return degraded.clamp(0, 1), strength


def _apply_snow(
    image: torch.Tensor,
    generator: torch.Generator,
) -> tuple[torch.Tensor, float]:
    _, height, width = image.shape
    strength = _uniform(generator, 0.20, 0.80)
    # Snow is a sparse occlusion. Keep the seed density low because pooling
    # expands every seed into a visible flake.
    density = 0.00015 + 0.0010 * strength
    impulses = (
        torch.rand((1, 1, height, width), generator=generator, dtype=image.dtype) < density
    ).to(image.dtype)
    small = F.max_pool2d(impulses, kernel_size=3, stride=1, padding=1)
    large = F.max_pool2d(impulses, kernel_size=5, stride=1, padding=2)
    flakes = (0.72 * small + 0.28 * large).clamp(0, 1)[0]
    alpha = flakes * (0.16 + 0.46 * strength)
    degraded = image * (1.0 - alpha) + alpha
    return degraded.clamp(0, 1), strength


def _apply_degradation(
    image: torch.Tensor,
    degradation: str,
    generator: torch.Generator,
) -> tuple[torch.Tensor, float]:
    if degradation in {"dust", "sand"}:
        return _apply_dust_or_sand(image, degradation, generator)
    if degradation == "haze":
        return _apply_haze(image, generator)
    if degradation == "lowlight":
        return _apply_lowlight(image, generator)
    if degradation == "colorcast":
        return _apply_colorcast(image, generator)
    if degradation == "rain":
        return _apply_rain(image, generator)
    if degradation == "snow":
        return _apply_snow(image, generator)
    raise ValueError(f"Unsupported synthetic degradation: {degradation}")


def synthesize_composite_degradation(
    clean: torch.Tensor,
    degradations: list[str] | tuple[str, ...],
    generator: torch.Generator,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Apply dense degradations first and sparse occlusions last.

    Returns the degraded CHW tensor and continuous labels in [0, 1].
    """
    requested = list(dict.fromkeys(degradations))
    unknown = sorted(set(requested) - set(DENSE_KEYS) - set(SPARSE_KEYS))
    if unknown:
        raise ValueError(f"Unknown synthetic degradations: {unknown}")
    strengths = {key: 0.0 for key in (*DENSE_KEYS, *SPARSE_KEYS)}
    degraded = clean
    for degradation in (*DENSE_KEYS, *SPARSE_KEYS):
        if degradation not in requested:
            continue
        degraded, strength = _apply_degradation(degraded, degradation, generator)
        strengths[degradation] = strength
    return degraded.clamp(0, 1), strengths


def synthesize_degradation(
    clean: torch.Tensor,
    degradation: str,
    generator: torch.Generator,
) -> torch.Tensor:
    """Backward-compatible single-degradation wrapper."""
    degraded, _ = synthesize_composite_degradation(clean, [degradation], generator)
    return degraded


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
        self.mixtures = [tuple(mixture) for mixture in config.get("mixtures", [])]
        for mixture in self.mixtures:
            _label_vector(list(mixture))
        self.dense_label, self.sparse_label = _label_vector(self.tasks)
        self.crop_size = crop_size
        self.training = training
        self.seed = seed
        self.synthetic = config.get("synthetic")
        self.clean_source = bool(self.synthetic or self.mixtures)
        self.metadata: dict[str, dict] = {}

        if self.clean_source:
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
            metadata_path = config.get("metadata_path")
            if metadata_path:
                metadata_file = Path(metadata_path)
                if not metadata_file.exists():
                    raise FileNotFoundError(
                        f"Metadata file not found for {self.name}: {metadata_file}"
                    )
                metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
                self.metadata = metadata.get("samples", metadata)
            if config.get("require_metadata", False):
                missing_metadata = [
                    input_path.name
                    for input_path, _ in self.pairs
                    if input_path.name not in self.metadata
                ]
                if missing_metadata:
                    preview = "\n".join(f"  {name}" for name in missing_metadata[:5])
                    raise FileNotFoundError(
                        f"{self.name} has {len(missing_metadata)} samples without metadata. "
                        f"Examples:\n{preview}"
                    )

            include_groups = {str(value) for value in config.get("include_groups", [])}
            exclude_groups = {str(value) for value in config.get("exclude_groups", [])}
            min_degradations = config.get("min_degradations")
            max_degradations = config.get("max_degradations")
            has_metadata_filter = bool(include_groups or exclude_groups) or any(
                value is not None for value in (min_degradations, max_degradations)
            )
            if has_metadata_filter:
                if not metadata_path:
                    raise ValueError(
                        f"{self.name} uses metadata filters but has no metadata_path"
                    )
                if min_degradations is not None and int(min_degradations) < 1:
                    raise ValueError("min_degradations must be at least 1")
                if max_degradations is not None and int(max_degradations) < 1:
                    raise ValueError("max_degradations must be at least 1")
                if (
                    min_degradations is not None
                    and max_degradations is not None
                    and int(min_degradations) > int(max_degradations)
                ):
                    raise ValueError("min_degradations cannot exceed max_degradations")

                filtered_pairs = []
                for pair in self.pairs:
                    record = self.metadata.get(pair[0].name, {})
                    group = str(record.get("group", ""))
                    tasks = record.get("tasks") or []
                    task_count = len(tasks)
                    if include_groups and group not in include_groups:
                        continue
                    if exclude_groups and group in exclude_groups:
                        continue
                    if min_degradations is not None and task_count < int(min_degradations):
                        continue
                    if max_degradations is not None and task_count > int(max_degradations):
                        continue
                    filtered_pairs.append(pair)
                self.pairs = filtered_pairs

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

        active_tasks = self.tasks
        dense_label = self.dense_label
        sparse_label = self.sparse_label
        metadata = self.metadata.get(input_path.name, {})
        if metadata:
            strengths = metadata.get("strengths", metadata)
            dense_label, sparse_label = _strength_vector(strengths)
            active_tasks = metadata.get("tasks") or [
                key
                for key in (*DENSE_KEYS, *SPARSE_KEYS)
                if float(strengths.get(key, 0.0)) > 0
            ]
        elif self.clean_source:
            if self.training:
                # DataLoader seeds each worker's global torch RNG. Drawing the
                # per-sample seed from it keeps augmentation random and reproducible.
                sample_seed = int(torch.randint(0, 2**63 - 1, ()).item())
                generator = torch.Generator().manual_seed(sample_seed)
            else:
                generator = torch.Generator().manual_seed(
                    _stable_seed(self.name, input_path.name, base_seed=self.seed)
                )
            if self.mixtures:
                mixture_index = int(
                    torch.randint(0, len(self.mixtures), (), generator=generator).item()
                )
                active_tasks = list(self.mixtures[mixture_index])
            else:
                active_tasks = [self.synthetic]
            image, strengths = synthesize_composite_degradation(
                target, active_tasks, generator
            )
            dense_label, sparse_label = _strength_vector(strengths)

        return {
            "input": image,
            "gt": target,
            "dense_label": dense_label.clone(),
            "sparse_label": sparse_label.clone(),
            "name": input_path.name,
            "source": metadata.get("category", self.name),
            "task": "+".join(active_tasks),
        }

    def _paired_transform(self, image: Image.Image, target: Image.Image) -> tuple[torch.Tensor, torch.Tensor]:
        if image.size != target.size:
            raise ValueError(
                f"{self.name}: input/GT size mismatch: input={image.size}, gt={target.size}"
            )
        if self.crop_size:
            width, height = image.size
            crop = self.crop_size
            if min(width, height) < crop:
                scale = crop / min(width, height)
                size = [max(crop, round(height * scale)), max(crop, round(width * scale))]
                image = TF.resize(image, size, interpolation=InterpolationMode.BICUBIC, antialias=True)
                target = TF.resize(target, size, interpolation=InterpolationMode.BICUBIC, antialias=True)
                width, height = image.size
            if self.training:
                left = random.randint(0, max(0, width - crop))
                top = random.randint(0, max(0, height - crop))
            else:
                # A fixed center crop makes frequent validation reproducible and
                # prevents large 1K/2K images from dominating wall-clock time.
                left = max(0, (width - crop) // 2)
                top = max(0, (height - crop) // 2)
            image = TF.crop(image, top, left, crop, crop)
            target = TF.crop(target, top, left, crop, crop)
            if self.training:
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
