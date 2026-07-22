import torch
from torch import nn


class SimpleDenseSparseFusion(nn.Module):
    """V1 fusion whose learned gate controls every expert path."""

    def __init__(self, channels: int, token_dim: int):
        super().__init__()
        self.alpha = nn.Sequential(
            nn.LayerNorm(token_dim * 2),
            nn.Linear(token_dim * 2, 1),
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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        token = torch.cat([dense_context, sparse_context], dim=1)
        alpha = self.alpha(token)[:, :, None, None]
        weighted_dense = alpha * dense_feature
        weighted_sparse = (1.0 - alpha) * sparse_feature
        fused = shared + self.fuse(torch.cat([shared, weighted_dense, weighted_sparse], dim=1))
        return fused, alpha
