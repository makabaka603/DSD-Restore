import torch
from torch import nn
import torch.nn.functional as F

from .dense_expert import pool_tokens


class HighFrequencyBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.laplace = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
        kernel = torch.tensor([[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]])
        self.laplace.weight.data.copy_(kernel.view(1, 1, 3, 3).repeat(channels, 1, 1, 1))
        self.laplace.weight.requires_grad_(False)
        self.mix = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )
        self.norm = nn.GroupNorm(8 if channels >= 8 else 1, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hf = self.laplace(x)
        y = self.mix(torch.cat([x, hf], dim=1))
        return self.norm(x + y)


class SparseOcclusionExpert(nn.Module):
    """Sparse expert for rain, raindrop, snow, and local occlusion."""

    def __init__(
        self,
        channels: int,
        token_dim: int,
        num_blocks: int = 4,
        token_pooling: str = "mean",
    ):
        super().__init__()
        token_pooling = token_pooling.lower()
        if token_pooling not in {"mean", "presence_weighted"}:
            raise ValueError("token_pooling must be 'mean' or 'presence_weighted'")
        self.token_pooling = token_pooling
        self.token_gate = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, channels),
            nn.Sigmoid(),
        )
        self.mask_predictor = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(channels, 1, 3, padding=1),
            nn.Sigmoid(),
        )
        self.directional = nn.Sequential(
            nn.Conv2d(channels, channels, (1, 7), padding=(0, 3), groups=channels),
            nn.Conv2d(channels, channels, (7, 1), padding=(3, 0), groups=channels),
            nn.Conv2d(channels, channels, 1),
            nn.GELU(),
        )
        self.edge_attention = nn.Sequential(nn.Conv2d(channels, 1, 3, padding=1), nn.Sigmoid())
        self.blocks = nn.Sequential(*[HighFrequencyBlock(channels) for _ in range(num_blocks)])

    def forward(
        self,
        feature: torch.Tensor,
        sparse_tokens: torch.Tensor,
        sparse_logits: torch.Tensor,
        prototype_context: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        token = (
            pool_tokens(sparse_tokens, sparse_logits, self.token_pooling)
            + prototype_context
        )
        gate = self.token_gate(token)[:, :, None, None]
        x = feature * gate
        mask = self.mask_predictor(x)
        streak_flake = self.directional(x)
        edge_gate = self.edge_attention(streak_flake)
        sparse_feature = self.blocks(feature + streak_flake * mask * edge_gate)
        return {
            "feature": sparse_feature,
            "mask": mask,
            "edge_gate": edge_gate,
        }
