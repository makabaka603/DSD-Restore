import random
from pathlib import Path

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(name: str = "auto") -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    moved = {}
    for key, value in batch.items():
        moved[key] = value.to(device, non_blocking=device.type == "cuda") if torch.is_tensor(value) else value
    return moved


def configure_cuda(device: torch.device, cudnn_benchmark: bool = True) -> None:
    if device.type != "cuda":
        return
    torch.backends.cudnn.benchmark = cudnn_benchmark
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def get_amp_settings(device: torch.device, enabled: bool, dtype_name: str = "bfloat16") -> tuple[bool, torch.dtype]:
    if not enabled or device.type != "cuda":
        return False, torch.float32
    if dtype_name.lower() in {"bfloat16", "bf16"}:
        if torch.cuda.is_bf16_supported():
            return True, torch.bfloat16
        print("BF16 is not supported on this GPU; falling back to FP16 AMP.")
    return True, torch.float16
