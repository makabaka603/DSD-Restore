import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from losses import DSDRestoreV1Loss
from metrics import gradient_global_norm, model_diagnostics, restoration_panel
from models import DSDRestoreV1


def main() -> None:
    model = DSDRestoreV1(base_channels=16, token_dim=32)
    batch = {
        "input": torch.rand(2, 3, 64, 64),
        "gt": torch.rand(2, 3, 64, 64),
        "dense_label": torch.tensor([[1, 0, 1, 0, 0], [0, 1, 0, 1, 0]], dtype=torch.float32),
        "sparse_label": torch.tensor([[1, 0, 0, 0], [0, 0, 1, 0]], dtype=torch.float32),
    }
    outputs = model(batch["input"])
    loss = DSDRestoreV1Loss()(outputs, batch)
    loss["total"].backward()
    diagnostics = model_diagnostics(outputs, batch)
    diagnostics["system/gradient_global_norm"] = gradient_global_norm(model)
    panel = restoration_panel(batch, outputs, max_samples=2)
    missing_gradients = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and parameter.grad is None
    ]
    if missing_gradients:
        raise RuntimeError(f"Trainable parameters without gradients: {missing_gradients}")
    if outputs["dense_prototype_weights"].shape != (2, 5):
        raise ValueError("Unexpected dense prototype weight shape")
    if outputs["sparse_prototype_weights"].shape != (2, 4):
        raise ValueError("Unexpected sparse prototype weight shape")
    required_diagnostics = {
        "prototype/dense/entropy",
        "prototype/sparse/offdiag_cosine",
        "routing/fusion_alpha_mean",
        "expert/dense_high_frequency_energy",
        "mask/mean_coverage",
        "tokenizer/macro_f1",
        "system/gradient_global_norm",
    }
    missing_diagnostics = required_diagnostics - diagnostics.keys()
    if missing_diagnostics:
        raise RuntimeError(f"Missing training diagnostics: {sorted(missing_diagnostics)}")
    if panel.shape != (3, 128, 384):
        raise ValueError(f"Unexpected TensorBoard panel shape: {tuple(panel.shape)}")
    print("DSD-Restore V1 smoke test passed")
    print(f"restored: {tuple(outputs['restored'].shape)}")
    print(f"dense_tokens: {tuple(outputs['dense_tokens'].shape)}")
    print(f"sparse_tokens: {tuple(outputs['sparse_tokens'].shape)}")
    print(f"prototype loss: {float(loss['proto']):.4f}")
    print(f"sparse-mask loss: {float(loss['sparse']):.4f}")
    print(f"diagnostic scalars: {len(diagnostics)}")
    print(f"visual panel: {tuple(panel.shape)}")
    print(f"loss: {float(loss['total'].detach()):.4f}")


if __name__ == "__main__":
    main()
