import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from losses import DSDRestoreV1Loss
from models import build_model
from utils.config import load_config


FORMAL_CONFIGS = (
    "configs/train_naf32_stage1.yaml",
    "configs/train_naf32_stage2.yaml",
    "configs/train_naf32_stage3.yaml",
    "configs/train_naf52_stage1.yaml",
    "configs/train_naf52_stage2.yaml",
    "configs/train_naf52_stage3.yaml",
)

EXPECTED_STAGE_CHAIN = {
    "naf32_stage1": None,
    "naf32_stage2": "naf32_stage1",
    "naf32_stage3": "naf32_stage2",
    "naf52_stage1": None,
    "naf52_stage2": "naf52_stage1",
    "naf52_stage3": "naf52_stage2",
}


def main() -> None:
    torch.manual_seed(42)
    criterion = DSDRestoreV1Loss(
        lambda_rec=1.0,
        lambda_ssim=0.2,
        lambda_freq=0.05,
        lambda_color=0.05,
        lambda_cls=0.0,
        lambda_proto=0.0,
        lambda_sparse=0.0,
    )
    parameter_counts: dict[int, int] = {}
    for config_path in FORMAL_CONFIGS:
        if not Path(config_path).exists():
            raise FileNotFoundError(config_path)
        cfg = load_config(config_path)
        stage = cfg["stage"]
        stage_name = stage["name"]
        expected_parent = EXPECTED_STAGE_CHAIN[stage_name]
        if stage.get("init_from_stage") != expected_parent:
            raise AssertionError(
                f"{stage_name}: expected init_from_stage={expected_parent!r}, "
                f"got {stage.get('init_from_stage')!r}"
            )
        model = build_model(cfg["model"])
        width = int(cfg["model"]["base_channels"])
        parameter_count = sum(p.numel() for p in model.parameters())
        previous_count = parameter_counts.setdefault(width, parameter_count)
        if previous_count != parameter_count:
            raise AssertionError(
                f"NAF{width} parameter count changed across stages: "
                f"{previous_count} vs {parameter_count}"
            )
        if getattr(model, "supports_dsd_diagnostics", True):
            raise AssertionError(f"{stage_name} unexpectedly enables DSD diagnostics")

    for width in sorted(parameter_counts):
        cfg = load_config(f"configs/train_naf{width}_stage1.yaml")
        model = build_model(cfg["model"]).train()
        image = torch.rand(1, 3, 32, 32)
        batch = {"gt": torch.rand_like(image)}
        outputs = model(image)
        if set(outputs) != {"restored"}:
            raise AssertionError(f"NAF{width} output keys: {sorted(outputs)}")
        losses = criterion(outputs, batch)
        losses["total"].backward()
        trainable = sum(parameter.numel() for parameter in model.parameters())
        with_grad = sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.grad is not None
        )
        if trainable != with_grad:
            raise RuntimeError(
                f"NAF{width}: parameters without gradients: {trainable - with_grad}"
            )
        print(
            f"NAF{width}: params={parameter_counts[width] / 1e6:.6f}M "
            f"output={tuple(outputs['restored'].shape)} "
            f"loss={float(losses['total'].detach()):.6f}"
        )

    print("NAF baseline smoke test passed")


if __name__ == "__main__":
    main()
