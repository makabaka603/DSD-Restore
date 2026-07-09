import torch
from torch import nn


class SimpleDenseSparseFusion(nn.Module):
    """V1 fusion: concat + 1x1 conv with learned dense/sparse mixing."""

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
        dense_tokens: torch.Tensor,
        sparse_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        token = torch.cat([dense_tokens.mean(dim=1), sparse_tokens.mean(dim=1)], dim=1)
        alpha = self.alpha(token)[:, :, None, None]
        mixed = alpha * dense_feature + (1.0 - alpha) * sparse_feature
        fused = self.fuse(torch.cat([shared, mixed, dense_feature + sparse_feature], dim=1))
        return fused, alpha
