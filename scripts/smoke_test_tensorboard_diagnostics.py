import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from losses import DSDRestoreV1Loss
from metrics import model_diagnostics, restoration_panel
from models import DSDRestoreV1
from train import flush_train_logs, write_prototype_histograms


class RecordingWriter:
    def __init__(self) -> None:
        self.scalars: set[str] = set()
        self.histograms: set[str] = set()
        self.images: set[str] = set()

    def add_scalar(self, name: str, value, step: int) -> None:
        del value, step
        self.scalars.add(name)

    def add_histogram(self, name: str, value, step: int) -> None:
        del value, step
        self.histograms.add(name)

    def add_image(self, name: str, value, step: int) -> None:
        del value, step
        self.images.add(name)


def main() -> None:
    model = DSDRestoreV1(base_channels=16, token_dim=32)
    criterion = DSDRestoreV1Loss()
    batch = {
        "input": torch.rand(2, 3, 64, 64),
        "gt": torch.rand(2, 3, 64, 64),
        "dense_label": torch.tensor(
            [[1, 0, 1, 0, 0], [0, 1, 0, 1, 0]], dtype=torch.float32
        ),
        "sparse_label": torch.tensor(
            [[1, 0, 0, 0], [0, 0, 1, 0]], dtype=torch.float32
        ),
        "task": ["dust+haze+rain", "sand+lowlight+snow"],
    }
    outputs = model(batch["input"])
    losses = criterion(outputs, batch)
    losses["total"].backward()
    diagnostics = model_diagnostics(outputs, batch)
    writer = RecordingWriter()
    records = [
        {
            "step": 20,
            "loss": losses["total"].detach(),
            "losses": {
                name: value.detach() for name, value in losses.items() if name != "total"
            },
            "weighted_losses": {
                name: value.detach()
                for name, value in criterion.weighted_components(losses).items()
            },
            "diagnostics": diagnostics,
            "lr": 2e-4,
            "metrics": {"psnr": torch.tensor(20.0), "ssim": torch.tensor(0.8)},
            "data_wait_s": 0.001,
            "train_start": None,
            "train_end": None,
            "metric_end": None,
        }
    ]
    running_best = {
        "psnr": -float("inf"),
        "ssim": -float("inf"),
        "lpips": float("inf"),
        "dists": float("inf"),
    }
    flush_train_logs(
        records,
        writer,
        running_best,
        torch.device("cpu"),
        batch_size=2,
        interval_started=time.perf_counter() - 0.1,
    )
    write_prototype_histograms(writer, outputs, step=20)
    writer.add_image(
        "validation_images/fixed_panel", restoration_panel(batch, outputs), step=20
    )

    required_scalars = {
        "train/loss",
        "train/loss_components/proto",
        "train/loss_weighted/sparse",
        "train_diagnostics/prototype/dense/entropy",
        "train_diagnostics/routing/fusion_alpha_mean",
        "train_diagnostics/tokenizer/macro_f1",
        "timing/iteration_wall_ms",
    }
    missing = required_scalars - writer.scalars
    if missing:
        raise RuntimeError(f"Missing TensorBoard scalar tags: {sorted(missing)}")
    if len(writer.histograms) != 9:
        raise RuntimeError(f"Expected 9 prototype histograms, got {len(writer.histograms)}")
    if "validation_images/fixed_panel" not in writer.images:
        raise RuntimeError("The fixed validation panel was not written")
    print("TensorBoard diagnostics smoke test passed")
    print(f"scalar tags: {len(writer.scalars)}")
    print(f"histogram tags: {len(writer.histograms)}")
    print(f"image tags: {len(writer.images)}")


if __name__ == "__main__":
    main()
