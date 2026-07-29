import torch
from torch import nn


class SimpleDenseSparseFusion(nn.Module):
    """Fuse dense and sparse experts with legacy or independent gates."""

    def __init__(
        self,
        channels: int,
        token_dim: int,
        gate_mode: str = "complementary",
    ):
        super().__init__()
        gate_mode = gate_mode.lower()
        if gate_mode not in {"complementary", "independent"}:
            raise ValueError("gate_mode must be 'complementary' or 'independent'")
        self.gate_mode = gate_mode
        if gate_mode == "complementary":
            # Preserve the original module names so old V1 checkpoints remain
            # strictly loadable without state-dict translation.
            self.alpha = nn.Sequential(
                nn.LayerNorm(token_dim * 2),
                nn.Linear(token_dim * 2, 1),
                nn.Sigmoid(),
            )
        else:
            self.gates = nn.Sequential(
                nn.LayerNorm(token_dim * 2),
                nn.Linear(token_dim * 2, 2),
                nn.Sigmoid(),
            )
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 3, channels, 1),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(
        self,
        shared: torch.Tensor,
        dense_feature: torch.Tensor,
        sparse_feature: torch.Tensor,
        dense_context: torch.Tensor,
        sparse_context: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        token = torch.cat([dense_context, sparse_context], dim=1)
        if self.gate_mode == "complementary":
            dense_gate = self.alpha(token)[:, :, None, None]
            sparse_gate = 1.0 - dense_gate
        else:
            gates = self.gates(token)
            dense_gate = gates[:, 0:1, None, None]
            sparse_gate = gates[:, 1:2, None, None]
        weighted_dense = dense_gate * dense_feature
        weighted_sparse = sparse_gate * sparse_feature
        fused = shared + self.fuse(torch.cat([shared, weighted_dense, weighted_sparse], dim=1))
        return fused, dense_gate, sparse_gate
