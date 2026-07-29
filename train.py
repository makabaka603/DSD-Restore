import argparse
import math
import os
import time
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, default_collate
from torchvision.transforms import functional as TF
from tqdm import tqdm

from datasets import PairedRestorationDataset, build_balanced_sampler, build_source_dataset
from losses import DSDRestoreV1Loss
from metrics import (
    DENSE_NAMES,
    SPARSE_NAMES,
    MultilabelAccumulator,
    PerceptualMetrics,
    batch_psnr_tensor,
    batch_ssim_tensor,
    gradient_global_norm,
    model_diagnostics,
    restoration_panel,
)
from models import build_model
from utils.config import load_config
from utils.runtime import (
    configure_cuda,
    ensure_dir,
    get_amp_settings,
    get_device,
    move_batch_to_device,
    seed_everything,
)


def build_loaders(cfg: dict) -> tuple[DataLoader, DataLoader]:
    data = cfg["data"]
    if data.get("train_sources"):
        seed = cfg["experiment"].get("seed", 42)
        train_set, train_sources = build_source_dataset(
            data["train_sources"],
            crop_size=data.get("crop_size"),
            training=True,
            seed=seed,
        )
        val_set, _ = build_source_dataset(
            data["val_sources"],
            crop_size=data.get("val_crop_size"),
            training=False,
            seed=seed,
        )
        samples_per_epoch = data.get("samples_per_epoch", data["batch_size"] * 1000)
        sampler = build_balanced_sampler(
            train_sources,
            data["train_sources"],
            num_samples=samples_per_epoch,
            seed=seed,
        )
        num_workers = data.get("num_workers", 0)
        loader_kwargs = {
            "batch_size": data["batch_size"],
            "sampler": sampler,
            "num_workers": num_workers,
            "pin_memory": True,
            "persistent_workers": data.get("persistent_workers", True) and num_workers > 0,
            "drop_last": True,
        }
        if num_workers > 0:
            loader_kwargs["prefetch_factor"] = data.get("prefetch_factor", 2)
        train_loader = DataLoader(train_set, **loader_kwargs)
        val_workers = data.get("val_num_workers", min(4, num_workers))
        val_kwargs = {
            "batch_size": 1,
            "shuffle": False,
            "num_workers": val_workers,
            "pin_memory": True,
            "persistent_workers": data.get("persistent_workers", True) and val_workers > 0,
        }
        if val_workers > 0:
            val_kwargs["prefetch_factor"] = data.get("val_prefetch_factor", 2)
        val_loader = DataLoader(val_set, **val_kwargs)
        print("Training sources:")
        for source, source_cfg in zip(train_sources, data["train_sources"]):
            print(
                f"  {source.name}: {len(source)} images, "
                f"probability={float(source_cfg['probability']):.1%}"
            )
        return train_loader, val_loader

    train_set = PairedRestorationDataset(
        data["train_input_dir"],
        data["train_gt_dir"],
        data.get("train_metadata"),
        crop_size=data.get("crop_size"),
        training=True,
    )
    val_set = PairedRestorationDataset(
        data["val_input_dir"],
        data["val_gt_dir"],
        data.get("val_metadata"),
        crop_size=data.get("val_crop_size"),
        training=False,
    )
    num_workers = data.get("num_workers", 0)
    loader_kwargs = {
        "batch_size": data["batch_size"],
        "shuffle": True,
        "num_workers": num_workers,
        "pin_memory": True,
        "persistent_workers": data.get("persistent_workers", True) and num_workers > 0,
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = data.get("prefetch_factor", 2)
    train_loader = DataLoader(
        train_set,
        **loader_kwargs,
    )
    val_workers = data.get("val_num_workers", min(4, num_workers))
    val_kwargs = {
        "batch_size": 1,
        "shuffle": False,
        "num_workers": val_workers,
        "pin_memory": True,
        "persistent_workers": data.get("persistent_workers", True) and val_workers > 0,
    }
    if val_workers > 0:
        val_kwargs["prefetch_factor"] = data.get("val_prefetch_factor", 2)
    val_loader = DataLoader(val_set, **val_kwargs)
    return train_loader, val_loader


def use_channels_last(batch: dict, enabled: bool) -> dict:
    if enabled:
        for key in ("input", "gt"):
            value = batch.get(key)
            if torch.is_tensor(value) and value.ndim == 4:
                batch[key] = value.contiguous(memory_format=torch.channels_last)
    return batch


@torch.no_grad()
def collect_fixed_visuals(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    channels_last: bool,
    max_samples: int,
) -> tuple[torch.Tensor | None, list[str]]:
    """Select deterministic, task-diverse samples across the full validation set."""
    dataset = loader.dataset
    if max_samples <= 0 or len(dataset) == 0:
        return None, []
    candidate_count = min(len(dataset), max(64, max_samples * 8))
    if candidate_count == 1:
        candidate_indices = [0]
    else:
        candidate_indices = [
            round(index * (len(dataset) - 1) / (candidate_count - 1))
            for index in range(candidate_count)
        ]
    rows: list[torch.Tensor] = []
    labels: list[str] = []
    seen_tasks: set[str] = set()
    fallback: list[tuple[dict, str]] = []
    for index in candidate_indices:
        sample = dataset[index]
        task = str(sample.get("task", "mixed"))
        if task in seen_tasks:
            if len(fallback) < max_samples:
                fallback.append((sample, task))
            continue
        visual_batch = default_collate([sample])
        visual_batch = use_channels_last(
            move_batch_to_device(visual_batch, device), channels_last
        )
        with torch.amp.autocast(
            device_type=device.type, enabled=amp_enabled, dtype=amp_dtype
        ):
            visual_outputs = model(visual_batch["input"])
        rows.append(restoration_panel(visual_batch, visual_outputs, max_samples=1))
        labels.append(task)
        seen_tasks.add(task)
        if len(rows) == max_samples:
            break
    for sample, task in fallback:
        if len(rows) == max_samples:
            break
        visual_batch = default_collate([sample])
        visual_batch = use_channels_last(
            move_batch_to_device(visual_batch, device), channels_last
        )
        with torch.amp.autocast(
            device_type=device.type, enabled=amp_enabled, dtype=amp_dtype
        ):
            visual_outputs = model(visual_batch["input"])
        rows.append(restoration_panel(visual_batch, visual_outputs, max_samples=1))
        labels.append(task)
    return (torch.cat(rows, dim=1), labels) if rows else (None, [])


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    perceptual: PerceptualMetrics,
    compute_perceptual: bool = True,
    perceptual_max_samples_per_source: int = 0,
    perceptual_max_size: int | None = None,
    max_samples_per_source: int = 0,
    channels_last: bool = False,
    collect_diagnostics: bool = True,
    visual_samples: int = 0,
) -> dict:
    model.eval()
    task_values: dict[str, dict[str, list[float]]] = {}
    perceptual_source_counts: dict[str, int] = {}
    source_counts: dict[str, int] = {}
    diagnostic_sums: dict[str, float] = {}
    diagnostic_counts: dict[str, int] = {}
    gate_moments = {
        "fusion_alpha": [0.0, 0.0, 0],
        "dense_gate": [0.0, 0.0, 0],
        "sparse_gate": [0.0, 0.0, 0],
    }
    tokenizer_stats = MultilabelAccumulator()
    for batch in loader:
        task = batch.get("task", ["mixed"])[0]
        source = batch.get("source", [task])[0]
        source_count = source_counts.get(source, 0)
        if max_samples_per_source > 0 and source_count >= max_samples_per_source:
            continue
        source_counts[source] = source_count + 1
        batch = use_channels_last(move_batch_to_device(batch, device), channels_last)
        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled, dtype=amp_dtype):
            outputs = model(batch["input"])
        if collect_diagnostics:
            diagnostics = model_diagnostics(outputs, batch, include_tokenizer=False)
            tokenizer_stats.update(outputs, batch)
            gate_outputs = {
                "fusion_alpha": outputs["fusion_alpha"],
                "dense_gate": outputs.get(
                    "fusion_dense_gate", outputs["fusion_alpha"]
                ),
                "sparse_gate": outputs.get(
                    "fusion_sparse_gate", 1.0 - outputs["fusion_alpha"]
                ),
            }
            for name, gate in gate_outputs.items():
                values = gate.detach().float().flatten()
                gate_moments[name][0] += float(values.sum().item())
                gate_moments[name][1] += float(values.square().sum().item())
                gate_moments[name][2] += values.numel()
            for name, raw_value in diagnostics.items():
                value = float(raw_value.item())
                diagnostic_sums[name] = diagnostic_sums.get(name, 0.0) + value
                diagnostic_counts[name] = diagnostic_counts.get(name, 0) + 1
        values = task_values.setdefault(task, {"psnr": [], "ssim": [], "lpips": [], "dists": []})
        values["psnr"].append(batch_psnr_tensor(outputs["restored"], batch["gt"]))
        values["ssim"].append(batch_ssim_tensor(outputs["restored"], batch["gt"]))
        source_count = perceptual_source_counts.get(source, 0)
        should_measure_perceptual = compute_perceptual and (
            perceptual_max_samples_per_source <= 0
            or source_count < perceptual_max_samples_per_source
        )
        if should_measure_perceptual:
            perceptual_values = perceptual(
                outputs["restored"], batch["gt"], max_size=perceptual_max_size
            )
            values["lpips"].append(perceptual_values["lpips"])
            values["dists"].append(perceptual_values["dists"])
            perceptual_source_counts[source] = source_count + 1

    per_task = {}
    for task, values in task_values.items():
        task_metrics = {
            "psnr": float(torch.stack(values["psnr"]).mean().item()),
            "ssim": float(torch.stack(values["ssim"]).mean().item()),
            "count": len(values["psnr"]),
        }
        if values["lpips"]:
            task_metrics["lpips"] = sum(values["lpips"]) / len(values["lpips"])
            task_metrics["dists"] = sum(values["dists"]) / len(values["dists"])
            task_metrics["perceptual_count"] = len(values["lpips"])
        per_task[task] = task_metrics

    macro_psnr = sum(values["psnr"] for values in per_task.values()) / len(per_task)
    macro_ssim = sum(values["ssim"] for values in per_task.values()) / len(per_task)
    result = {
        "psnr": macro_psnr,
        "ssim": macro_ssim,
        "per_task": per_task,
    }
    perceptual_tasks = [values for values in per_task.values() if "lpips" in values]
    if perceptual_tasks:
        result["lpips"] = sum(values["lpips"] for values in perceptual_tasks) / len(perceptual_tasks)
        result["dists"] = sum(values["dists"] for values in perceptual_tasks) / len(perceptual_tasks)
    if collect_diagnostics:
        diagnostics = {
            name: diagnostic_sums[name] / diagnostic_counts[name]
            for name in diagnostic_sums
        }
        for name, (total, squared_total, count) in gate_moments.items():
            if count <= 0:
                continue
            mean = total / count
            variance = max(0.0, squared_total / count - mean * mean)
            diagnostics[f"routing/{name}_mean"] = mean
            diagnostics[f"routing/{name}_std"] = math.sqrt(variance)
        diagnostics.update(
            {name: float(value.item()) for name, value in tokenizer_stats.compute().items()}
        )
        result["diagnostics"] = diagnostics
    if visual_samples > 0:
        visual_panel, visual_labels = collect_fixed_visuals(
            model,
            loader,
            device,
            amp_enabled,
            amp_dtype,
            channels_last,
            visual_samples,
        )
        if visual_panel is not None:
            result["visual_panel"] = visual_panel
            result["visual_labels"] = visual_labels
    return result


def prepare_tensorboard_dir(cfg: dict) -> Path:
    """Create /root/tf-logs as a symlink so events live on the data disk."""
    log_dir = Path(cfg.get("log_dir", "/root/tf-logs"))
    storage_dir = Path(cfg.get("log_storage_dir", "/root/autodl-tmp/tf-logs"))
    use_symlink = cfg.get("symlink_log_dir", True)
    if use_symlink:
        storage_dir.mkdir(parents=True, exist_ok=True)
        if log_dir.is_symlink():
            if log_dir.resolve() != storage_dir.resolve():
                raise RuntimeError(f"{log_dir} points to {log_dir.resolve()}, not {storage_dir}")
        elif log_dir.exists():
            raise RuntimeError(
                f"{log_dir} already exists and is not a symlink. Move it to {storage_dir}, "
                f"then run: ln -s {storage_dir} {log_dir}"
            )
        else:
            log_dir.parent.mkdir(parents=True, exist_ok=True)
            os.symlink(storage_dir, log_dir, target_is_directory=True)
    run_dir = log_dir / cfg.get("run_name", "default")
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def is_better(value: float, best: float, metric: str) -> bool:
    return value < best if metric in {"lpips", "dists"} else value > best


def write_metrics(writer: object, prefix: str, metrics: dict, step: int) -> None:
    for name in ("psnr", "ssim", "lpips", "dists"):
        if name in metrics:
            writer.add_scalar(f"{prefix}/{name}", metrics[name], step)


def write_scalar_dict(writer: object, prefix: str, values: dict, step: int) -> None:
    for name, raw_value in values.items():
        value = float(raw_value.item()) if torch.is_tensor(raw_value) else float(raw_value)
        writer.add_scalar(f"{prefix}/{name}", value, step)


def write_prototype_histograms(writer: object, outputs: dict, step: int) -> None:
    dense_weights = outputs["dense_prototype_weights"].detach().float().cpu()
    sparse_weights = outputs["sparse_prototype_weights"].detach().float().cpu()
    for index, name in enumerate(DENSE_NAMES):
        writer.add_histogram(f"prototype_hist/dense/{name}", dense_weights[:, index], step)
    for index, name in enumerate(SPARSE_NAMES):
        writer.add_histogram(f"prototype_hist/sparse/{name}", sparse_weights[:, index], step)


def flush_train_logs(
    records: list[dict],
    writer: object,
    running_best: dict[str, float],
    device: torch.device,
    batch_size: int,
    interval_started: float,
) -> tuple[float, dict[str, float]]:
    """Synchronize once, then emit every buffered per-iteration TensorBoard point."""
    if not records:
        return interval_started, {}
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = max(time.perf_counter() - interval_started, 1e-9)
    latest: dict[str, float] = {}
    train_gpu_ms = []
    metric_gpu_ms = []
    for record in records:
        step = record["step"]
        loss = float(record["loss"].item())
        latest["loss"] = loss
        writer.add_scalar("train/loss", loss, step)
        writer.add_scalar("train/learning_rate", record["lr"], step)
        write_scalar_dict(writer, "train/loss_components", record["losses"], step)
        write_scalar_dict(writer, "train/loss_weighted", record["weighted_losses"], step)
        if record.get("diagnostics"):
            write_scalar_dict(writer, "train_diagnostics", record["diagnostics"], step)
        for name, raw_value in record["metrics"].items():
            value = float(raw_value.item()) if torch.is_tensor(raw_value) else float(raw_value)
            latest[name] = value
            writer.add_scalar(f"train/current/{name}", value, step)
            if is_better(value, running_best[name], name):
                running_best[name] = value
            writer.add_scalar(f"train/best/{name}", running_best[name], step)
        if record.get("train_start") is not None:
            train_gpu_ms.append(record["train_start"].elapsed_time(record["train_end"]))
            metric_gpu_ms.append(record["train_end"].elapsed_time(record["metric_end"]))

    last_step = records[-1]["step"]
    writer.add_scalar(
        "timing/data_wait_ms",
        1000.0 * sum(record["data_wait_s"] for record in records) / len(records),
        last_step,
    )
    writer.add_scalar("timing/interval_wall_s", elapsed, last_step)
    writer.add_scalar("timing/iteration_wall_ms", 1000.0 * elapsed / len(records), last_step)
    writer.add_scalar(
        "timing/train_images_per_s", len(records) * batch_size / elapsed, last_step
    )
    if train_gpu_ms:
        writer.add_scalar("timing/train_gpu_ms", sum(train_gpu_ms) / len(train_gpu_ms), last_step)
        writer.add_scalar("timing/metric_gpu_ms", sum(metric_gpu_ms) / len(metric_gpu_ms), last_step)
    records.clear()
    return time.perf_counter(), latest


def build_scheduler(optim: AdamW, max_iters: int, warmup_iters: int, min_lr: float) -> LambdaLR:
    base_lr = optim.param_groups[0]["lr"]
    min_ratio = min_lr / base_lr

    def lr_multiplier(step: int) -> float:
        if warmup_iters > 0 and step < warmup_iters:
            return max(1, step + 1) / warmup_iters
        progress = (step - warmup_iters) / max(1, max_iters - warmup_iters)
        progress = min(max(progress, 0.0), 1.0)
        return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optim, lr_lambda=lr_multiplier)


def stage_name(config: dict) -> str | None:
    stage = config.get("stage", {})
    return stage.get("name") if isinstance(stage, dict) else None


def checkpoint_stage_name(checkpoint: dict) -> str | None:
    checkpoint_config = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
    return stage_name(checkpoint_config) if isinstance(checkpoint_config, dict) else None


def load_initial_model_weights(
    model: torch.nn.Module,
    checkpoint_path: str | Path,
    config: dict,
) -> str | None:
    """Initialize a new stage from model weights without optimizer state."""
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Initial checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location="cpu")
    state_dict = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    if not isinstance(state_dict, dict):
        raise TypeError(f"Initial checkpoint does not contain a model state dict: {path}")

    expected_stage = config.get("stage", {}).get("init_from_stage")
    source_stage = checkpoint_stage_name(checkpoint)
    if expected_stage and source_stage != expected_stage:
        found = source_stage or "unlabeled checkpoint"
        raise ValueError(
            f"Stage {stage_name(config)} must initialize from {expected_stage}, found {found}: {path}"
        )
    init_strict = bool(config.get("runtime", {}).get("init_strict", True))
    if init_strict:
        model.load_state_dict(state_dict, strict=True)
    else:
        model_state = model.state_dict()
        compatible = {
            name: value
            for name, value in state_dict.items()
            if name in model_state and model_state[name].shape == value.shape
        }
        total_numel = sum(value.numel() for value in model_state.values())
        loaded_numel = sum(model_state[name].numel() for name in compatible)
        coverage = loaded_numel / max(total_numel, 1)
        min_coverage = float(
            config.get("runtime", {}).get("init_min_coverage", 0.95)
        )
        if coverage < min_coverage:
            raise RuntimeError(
                f"Compatible initialization coverage {coverage:.2%} is below "
                f"runtime.init_min_coverage={min_coverage:.2%}: {path}"
            )
        incompatible = model.load_state_dict(compatible, strict=False)
        shape_mismatches = sorted(
            name
            for name, value in state_dict.items()
            if name in model_state and model_state[name].shape != value.shape
        )
        print(
            f"Compatible checkpoint initialization loaded {coverage:.2%} of model "
            f"state by element count; missing={len(incompatible.missing_keys)}, "
            f"unexpected={len(incompatible.unexpected_keys)}, "
            f"shape_mismatches={len(shape_mismatches)}."
        )
        if shape_mismatches:
            print(f"  Reinitialized shape-mismatched keys: {shape_mismatches}")
    return source_stage


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optim: AdamW,
    scheduler: LambdaLR,
    scaler: torch.amp.GradScaler,
    cfg: dict,
    global_step: int,
    epoch: int,
    best_value: float,
    best_step: int,
    metrics: dict[str, float],
    running_best: dict[str, float] | None = None,
) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optim.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "global_step": global_step,
            "epoch": epoch,
            "best_value": best_value,
            # Kept for compatibility with checkpoints created by older code.
            "best_psnr": best_value if cfg["runtime"].get("selection_metric", "psnr") == "psnr" else None,
            "best_step": best_step,
            "metrics": metrics,
            "running_best": running_best,
            "config": cfg,
        },
        path,
    )


def save_weights(
    path: Path,
    model: torch.nn.Module,
    cfg: dict,
    global_step: int,
    metrics: dict[str, float],
) -> None:
    """Save a compact inference checkpoint without optimizer/scaler states."""
    torch.save(
        {
            "model": model.state_dict(),
            "global_step": global_step,
            "metrics": metrics,
            "config": cfg,
        },
        path,
    )


def train(
    cfg: dict,
    resume: str | None = None,
    init_checkpoint: str | None = None,
) -> None:
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError as exc:
        raise ImportError(
            "TensorBoard logging is enabled but tensorboard is not installed. "
            "Run: python -m pip install tensorboard"
        ) from exc

    seed_everything(cfg["experiment"].get("seed", 42))
    torch_home = Path(cfg["runtime"].get("torch_home", "/root/autodl-tmp/torch-cache"))
    torch_home.mkdir(parents=True, exist_ok=True)
    os.environ["TORCH_HOME"] = str(torch_home)
    device = get_device(cfg["runtime"].get("device", "auto"))
    configure_cuda(device, cfg["runtime"].get("cudnn_benchmark", True))
    amp_enabled, amp_dtype = get_amp_settings(
        device,
        cfg["runtime"].get("amp", False),
        cfg["runtime"].get("amp_dtype", "bfloat16"),
    )
    output_dir = ensure_dir(cfg["experiment"]["output_dir"])
    ckpt_dir = ensure_dir(cfg["runtime"].get("checkpoint_dir", output_dir / "checkpoints"))
    tensorboard_cfg = cfg["runtime"].get("tensorboard", {})
    tensorboard_dir = prepare_tensorboard_dir(tensorboard_cfg)
    train_loader, val_loader = build_loaders(cfg)

    metadata_path = cfg["data"].get("train_metadata")
    if not cfg["data"].get("train_sources") and cfg["loss"].get("lambda_cls", 0.0) > 0 and (
        not metadata_path or not Path(metadata_path).exists()
    ):
        raise FileNotFoundError(
            "lambda_cls is greater than zero, but train_metadata is missing. "
            "Provide degradation labels or set loss.lambda_cls to 0.0."
        )

    resume_path = resume or cfg["runtime"].get("resume")
    configured_init = init_checkpoint or cfg["runtime"].get("init_checkpoint")
    # Resuming restores the full state of the current stage. Its configured
    # cross-stage initializer must not be applied a second time.
    initial_path = None if resume_path else configured_init

    channels_last_enabled = cfg["runtime"].get("channels_last", False) and device.type == "cuda"
    model = build_model(cfg["model"])
    supports_dsd_diagnostics = bool(
        getattr(model, "supports_dsd_diagnostics", False)
    )
    if initial_path:
        source_stage = load_initial_model_weights(model, initial_path, cfg)
        print(
            f"Initialized {stage_name(cfg) or 'training'} from {initial_path} "
            f"(source stage: {source_stage or 'unlabeled'}); optimizer and iteration reset."
        )
    model = model.to(device)
    if channels_last_enabled:
        model = model.to(memory_format=torch.channels_last)
    perceptual = PerceptualMetrics(device)
    criterion = DSDRestoreV1Loss(**cfg["loss"])
    optim = AdamW(model.parameters(), lr=cfg["optimization"]["lr"], weight_decay=cfg["optimization"]["weight_decay"])
    max_iters = cfg["optimization"]["max_iters"]
    scheduler = build_scheduler(
        optim,
        max_iters=max_iters,
        warmup_iters=cfg["optimization"].get("warmup_iters", 0),
        min_lr=cfg["optimization"].get("min_lr", 1e-6),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled and amp_dtype == torch.float16)
    selection_metric = cfg["runtime"].get("selection_metric", "psnr").lower()
    if selection_metric not in {"psnr", "ssim", "lpips", "dists"}:
        raise ValueError("selection_metric must be one of: psnr, ssim, lpips, dists")
    best_value = math.inf if selection_metric in {"lpips", "dists"} else -math.inf
    best_step = 0
    global_step = 0
    epoch = 0
    last_metrics = {name: float("nan") for name in ("psnr", "ssim", "lpips", "dists")}
    running_best = {"psnr": -math.inf, "ssim": -math.inf, "lpips": math.inf, "dists": math.inf}

    if resume_path:
        checkpoint = torch.load(resume_path, map_location=device)
        current_stage = stage_name(cfg)
        resumed_stage = checkpoint_stage_name(checkpoint)
        if current_stage and resumed_stage and current_stage != resumed_stage:
            raise ValueError(
                f"Cannot resume {current_stage} from a {resumed_stage} checkpoint. "
                "Use --init-checkpoint for a stage transition."
            )
        model.load_state_dict(checkpoint["model"])
        optim.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        if checkpoint.get("scaler"):
            scaler.load_state_dict(checkpoint["scaler"])
        global_step = checkpoint.get("global_step", 0)
        epoch = checkpoint.get("epoch", 0)
        best_value = checkpoint.get("best_value")
        if best_value is None:
            best_value = checkpoint.get("best_psnr", checkpoint.get("metrics", {}).get(selection_metric, best_value))
        best_step = checkpoint.get("best_step", global_step)
        last_metrics = checkpoint.get("metrics", last_metrics)
        running_best.update(checkpoint.get("running_best") or {})
        print(f"Resumed from {resume_path} at iteration {global_step}.")

    active_model: torch.nn.Module = model
    compile_cfg = cfg["runtime"].get("compile", {})
    compile_enabled = compile_cfg.get("enabled", False) and device.type == "cuda"
    if compile_enabled:
        try:
            active_model = torch.compile(
                model,
                mode=compile_cfg.get("mode", "max-autotune-no-cudagraphs"),
                dynamic=compile_cfg.get("dynamic", False),
            )
            print(f"torch.compile enabled: mode={compile_cfg.get('mode', 'max-autotune-no-cudagraphs')}")
        except Exception as exc:
            active_model = model
            print(f"torch.compile setup failed; continuing in eager mode: {exc}")

    # purge_step prevents a resumed run from displaying stale duplicate points
    # at and after the first newly written iteration.
    writer = SummaryWriter(
        tensorboard_dir,
        purge_step=global_step + 1 if resume_path else None,
    )

    print(
        f"stage={stage_name(cfg) or 'unlabeled'} device={device} "
        f"amp={amp_enabled} amp_dtype={amp_dtype} "
        f"batch_size={cfg['data']['batch_size']} max_iters={max_iters} "
        f"channels_last={channels_last_enabled} compile={active_model is not model}"
    )

    val_every = cfg["runtime"].get("val_every_iters", 2000)
    save_every = cfg["runtime"].get("save_every_iters", 5000)
    window_every = cfg["runtime"].get("best_window_iters", 100)
    train_metrics_cfg = cfg["runtime"].get("train_metrics", {})
    psnr_ssim_every = train_metrics_cfg.get("psnr_ssim_every_iters", 1)
    perceptual_every = train_metrics_cfg.get("perceptual_every_iters", 100)
    perceptual_samples = train_metrics_cfg.get("perceptual_batch_samples", 4)
    val_perceptual_every = cfg["runtime"].get("val_perceptual_every_iters", 1000)
    val_perceptual_samples = cfg["runtime"].get("val_perceptual_max_samples_per_source", 2)
    val_perceptual_max_size = cfg["runtime"].get("val_perceptual_max_size", 256)
    selection_val_every = cfg["runtime"].get("selection_val_every_iters", 500)
    probe_val_samples = cfg["runtime"].get("probe_val_max_samples_per_source", 2)
    selection_val_samples = cfg["runtime"].get("selection_val_max_samples_per_source", 10)
    log_every = cfg["runtime"].get("log_every_iters", 20)
    early_stop_patience = cfg["runtime"].get("early_stop_patience_iters", 0)
    diagnostics_cfg = cfg["runtime"].get("diagnostics", {})
    diagnostics_every = diagnostics_cfg.get("scalar_every_iters", 20)
    histogram_every = diagnostics_cfg.get("histogram_every_iters", 500)
    image_every = diagnostics_cfg.get("image_every_iters", 1000)
    image_samples = diagnostics_cfg.get("image_samples", 8)
    save_image_panels = diagnostics_cfg.get("save_image_panels", True)
    if min(
        val_every, save_every, window_every, psnr_ssim_every, perceptual_every,
        perceptual_samples, val_perceptual_every, val_perceptual_samples,
        val_perceptual_max_size, selection_val_every, probe_val_samples,
        selection_val_samples, log_every, diagnostics_every, histogram_every,
        image_every, image_samples,
    ) <= 0:
        raise ValueError(
            "validation, save, metric, diagnostic, sample, and best-window values "
            "must be greater than zero."
        )
    if val_every > window_every or window_every % val_every != 0:
        raise ValueError(
            "best_window_iters must be an exact multiple of val_every_iters so every "
            "checkpoint window contains comparable validation measurements."
        )
    if selection_val_every % val_every != 0:
        raise ValueError("selection_val_every_iters must be a multiple of val_every_iters.")
    if selection_metric in {"lpips", "dists"} and val_perceptual_every != val_every:
        raise ValueError(
            "LPIPS/DISTS checkpoint selection requires val_perceptual_every_iters "
            "to equal val_every_iters."
        )
    stop_training = False
    window_best = math.inf if selection_metric in {"lpips", "dists"} else -math.inf
    active_window_start = ((global_step // window_every) * window_every) + 1
    log_records: list[dict] = []
    log_interval_started = time.perf_counter()
    loader_ready_at = time.perf_counter()
    compiled_step_verified = active_model is model
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    while global_step < max_iters and not stop_training:
        epoch += 1
        active_model.train()
        remaining = max_iters - global_step
        pbar = tqdm(train_loader, total=min(len(train_loader), remaining), desc=f"Epoch {epoch}")
        for batch in pbar:
            batch_received_at = time.perf_counter()
            data_wait_s = batch_received_at - loader_ready_at
            if global_step >= max_iters:
                break
            next_step = global_step + 1
            criterion.set_training_progress(next_step / max_iters)
            diagnostic_step = next_step % diagnostics_every == 0
            batch = use_channels_last(move_batch_to_device(batch, device), channels_last_enabled)
            train_start = train_end = metric_end = None
            if device.type == "cuda":
                train_start = torch.cuda.Event(enable_timing=True)
                train_end = torch.cuda.Event(enable_timing=True)
                metric_end = torch.cuda.Event(enable_timing=True)
                train_start.record()
            optim.zero_grad(set_to_none=True)
            try:
                with torch.amp.autocast(device_type=device.type, enabled=amp_enabled, dtype=amp_dtype):
                    outputs = active_model(batch["input"])
                    losses = criterion(outputs, batch)
                scaler.scale(losses["total"]).backward()
                compiled_step_verified = True
            except Exception as exc:
                if compiled_step_verified or active_model is model:
                    raise
                print(f"torch.compile first step failed; falling back to eager mode: {exc}")
                optim.zero_grad(set_to_none=True)
                active_model = model
                active_model.train()
                with torch.amp.autocast(device_type=device.type, enabled=amp_enabled, dtype=amp_dtype):
                    outputs = active_model(batch["input"])
                    losses = criterion(outputs, batch)
                scaler.scale(losses["total"]).backward()
                compiled_step_verified = True
            diagnostics = {}
            if diagnostic_step:
                if scaler.is_enabled():
                    scaler.unscale_(optim)
                with torch.no_grad():
                    if supports_dsd_diagnostics:
                        diagnostics.update(model_diagnostics(outputs, batch))
                    diagnostics["loss/auxiliary_scale"] = torch.tensor(
                        criterion.auxiliary_scale,
                        device=device,
                    )
                    diagnostics["system/gradient_global_norm"] = gradient_global_norm(model)
                    diagnostics["system/amp_scale"] = torch.tensor(
                        float(scaler.get_scale()), device=device
                    )
                    if device.type == "cuda":
                        gib = float(1024 ** 3)
                        diagnostics["system/gpu_memory_allocated_gb"] = torch.tensor(
                            torch.cuda.memory_allocated(device) / gib, device=device
                        )
                        diagnostics["system/gpu_memory_reserved_gb"] = torch.tensor(
                            torch.cuda.memory_reserved(device) / gib, device=device
                        )
                        diagnostics["system/gpu_memory_peak_gb"] = torch.tensor(
                            torch.cuda.max_memory_allocated(device) / gib, device=device
                        )
            scaler.step(optim)
            scaler.update()
            scheduler.step()
            if train_end is not None:
                train_end.record()
            global_step += 1
            restored = outputs["restored"].detach()
            if supports_dsd_diagnostics and global_step % histogram_every == 0:
                write_prototype_histograms(writer, outputs, global_step)
            if diagnostic_step and device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)

            train_metrics = {}
            if global_step % psnr_ssim_every == 0:
                train_metrics.update(
                    psnr=batch_psnr_tensor(restored, batch["gt"]),
                    ssim=batch_ssim_tensor(restored, batch["gt"]),
                )
            if global_step % perceptual_every == 0:
                sample_count = min(perceptual_samples, restored.shape[0])
                train_metrics.update(
                    perceptual(restored[:sample_count], batch["gt"][:sample_count])
                )
            if metric_end is not None:
                metric_end.record()
            log_records.append(
                {
                    "step": global_step,
                    "loss": losses["total"].detach(),
                    "losses": {
                        name: value.detach()
                        for name, value in losses.items()
                        if name != "total"
                    },
                    "weighted_losses": {
                        name: value.detach()
                        for name, value in criterion.weighted_components(losses).items()
                    },
                    "diagnostics": diagnostics,
                    "lr": optim.param_groups[0]["lr"],
                    "metrics": train_metrics,
                    "data_wait_s": data_wait_s,
                    "train_start": train_start,
                    "train_end": train_end,
                    "metric_end": metric_end,
                }
            )

            should_validate = global_step % val_every == 0 or global_step == max_iters
            should_flush = global_step % log_every == 0 or should_validate
            if should_flush:
                log_interval_started, latest_log = flush_train_logs(
                    log_records, writer, running_best, device,
                    cfg["data"]["batch_size"], log_interval_started,
                )
                pbar.set_postfix(
                    iteration=global_step,
                    loss=f"{latest_log.get('loss', float('nan')):.4f}",
                    lr=f"{optim.param_groups[0]['lr']:.2e}",
                )

            if should_validate:
                validation_started = time.perf_counter()
                measure_val_perceptual = global_step % val_perceptual_every == 0
                selection_validation = (
                    global_step % selection_val_every == 0 or global_step == max_iters
                )
                val_sample_limit = (
                    selection_val_samples if selection_validation else probe_val_samples
                )
                current_metrics = validate(
                    model, val_loader, device, amp_enabled, amp_dtype, perceptual,
                    compute_perceptual=measure_val_perceptual,
                    perceptual_max_samples_per_source=val_perceptual_samples,
                    perceptual_max_size=val_perceptual_max_size,
                    max_samples_per_source=val_sample_limit,
                    channels_last=channels_last_enabled,
                    collect_diagnostics=supports_dsd_diagnostics,
                    visual_samples=(
                        image_samples
                        if supports_dsd_diagnostics
                        and (global_step % image_every == 0 or global_step == max_iters)
                        else 0
                    ),
                )
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                validation_seconds = time.perf_counter() - validation_started
                writer.add_scalar("timing/validation_s", validation_seconds, global_step)
                writer.add_scalar("validation/sample_limit_per_source", val_sample_limit, global_step)
                val_prefix = "val" if selection_validation else "val_probe"
                task_prefix = "val_by_task" if selection_validation else "val_probe_by_task"
                validation_diagnostics = current_metrics.pop("diagnostics", {})
                visual_panel = current_metrics.pop("visual_panel", None)
                visual_labels = current_metrics.pop("visual_labels", [])
                write_metrics(writer, val_prefix, current_metrics, global_step)
                if validation_diagnostics:
                    write_scalar_dict(
                        writer,
                        f"{val_prefix}_diagnostics",
                        validation_diagnostics,
                        global_step,
                    )
                if visual_panel is not None:
                    writer.add_image("validation_images/fixed_panel", visual_panel, global_step)
                    if save_image_panels:
                        visualization_dir = ensure_dir(output_dir / "visualizations")
                        panel_path = visualization_dir / f"step_{global_step:07d}_panel.png"
                        TF.to_pil_image(visual_panel.clamp(0, 1)).save(panel_path)
                        panel_path.with_suffix(".txt").write_text(
                            "Columns: Input | Restored | GT | Absolute Error | Sparse Mask | "
                            f"Mask Overlay\nRows: {' | '.join(visual_labels)}\n",
                            encoding="utf-8",
                        )
                    writer.add_text(
                        "validation_images/layout",
                        "Columns: Input | Restored | GT | Absolute Error | Sparse Mask | "
                        f"Mask Overlay. Rows: {' | '.join(visual_labels)}",
                        global_step,
                    )
                for task, task_metrics in current_metrics.get("per_task", {}).items():
                    write_metrics(writer, f"{task_prefix}/{task}", task_metrics, global_step)
                message = (
                    f"iteration={global_step} validation={('selection' if selection_validation else 'probe')} "
                    f"macro_psnr={current_metrics['psnr']:.3f} "
                    f"macro_ssim={current_metrics['ssim']:.4f}"
                )
                if measure_val_perceptual:
                    message += (
                        f" macro_lpips={current_metrics['lpips']:.4f} "
                        f"macro_dists={current_metrics['dists']:.4f}"
                    )
                print(message)
                for task, task_metrics in current_metrics.get("per_task", {}).items():
                    task_message = (
                        f"  {task}: count={task_metrics['count']} "
                        f"psnr={task_metrics['psnr']:.3f} ssim={task_metrics['ssim']:.4f}"
                    )
                    if "lpips" in task_metrics:
                        task_message += (
                            f" lpips={task_metrics['lpips']:.4f} "
                            f"dists={task_metrics['dists']:.4f} "
                            f"perceptual_count={task_metrics['perceptual_count']}"
                        )
                    print(task_message)
                last_metrics = {**last_metrics, **current_metrics}
                if selection_validation and is_better(
                    current_metrics[selection_metric], best_value, selection_metric
                ):
                    best_value = current_metrics[selection_metric]
                    best_step = global_step
                    save_weights(ckpt_dir / "best.pth", model, cfg, global_step, last_metrics)
                elif (
                    selection_validation
                    and early_stop_patience > 0
                    and global_step - best_step >= early_stop_patience
                ):
                    print(
                        f"Early stopping at iteration {global_step}: validation {selection_metric} has not "
                        f"improved for {global_step - best_step} iterations."
                    )
                    stop_training = True

                # Keep overwriting this window's file only while its validation
                # selection metric improves. Thus val_every_iters may be smaller
                # than best_window_iters without producing redundant snapshots.
                window_start = ((global_step - 1) // window_every) * window_every + 1
                window_end = window_start + window_every - 1
                if window_start != active_window_start:
                    active_window_start = window_start
                    window_best = math.inf if selection_metric in {"lpips", "dists"} else -math.inf
                if is_better(last_metrics[selection_metric], window_best, selection_metric):
                    window_best = last_metrics[selection_metric]
                    save_weights(
                        ckpt_dir / f"window_{window_start:07d}_{window_end:07d}_best.pth",
                        model, cfg, global_step, last_metrics,
                    )
                active_model.train()
                loader_ready_at = time.perf_counter()
                log_interval_started = loader_ready_at

            if global_step % save_every == 0 or global_step == max_iters:
                checkpoint_started = time.perf_counter()
                save_checkpoint(
                    ckpt_dir / "latest.pth", model, optim, scheduler, scaler, cfg,
                    global_step, epoch, best_value, best_step, last_metrics, running_best,
                )
                save_checkpoint(
                    ckpt_dir / f"iter_{global_step:07d}.pth", model, optim, scheduler, scaler, cfg,
                    global_step, epoch, best_value, best_step, last_metrics, running_best,
                )
                checkpoint_finished = time.perf_counter()
                writer.add_scalar(
                    "timing/checkpoint_s", checkpoint_finished - checkpoint_started, global_step
                )
                loader_ready_at = checkpoint_finished
                log_interval_started = checkpoint_finished

            if stop_training:
                save_checkpoint(
                    ckpt_dir / "latest.pth", model, optim, scheduler, scaler, cfg,
                    global_step, epoch, best_value, best_step, last_metrics, running_best,
                )
                break
            loader_ready_at = time.perf_counter()
    writer.flush()
    writer.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_v1_stage1.yaml")
    checkpoint_group = parser.add_mutually_exclusive_group()
    checkpoint_group.add_argument(
        "--resume",
        default=None,
        help="resume the same stage with model, optimizer, scheduler, scaler, and iteration",
    )
    checkpoint_group.add_argument(
        "--init-checkpoint",
        default=None,
        help="start a new stage from model weights only; optimizer and iteration reset",
    )
    args = parser.parse_args()
    train(
        load_config(args.config),
        resume=args.resume,
        init_checkpoint=args.init_checkpoint,
    )


if __name__ == "__main__":
    main()
