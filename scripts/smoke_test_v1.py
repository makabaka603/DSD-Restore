import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from losses import DSDRestoreV1Loss
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
    print("DSD-Restore V1 smoke test passed")
    print(f"restored: {tuple(outputs['restored'].shape)}")
    print(f"dense_tokens: {tuple(outputs['dense_tokens'].shape)}")
    print(f"sparse_tokens: {tuple(outputs['sparse_tokens'].shape)}")
    print(f"loss: {float(loss['total'].detach()):.4f}")


if __name__ == "__main__":
    main()
