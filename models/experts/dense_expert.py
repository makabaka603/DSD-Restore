import torch
from torch import nn


class FiLM(nn.Module):
    def __init__(self, token_dim: int, channels: int):
        super().__init__()
        self.to_params = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, channels * 2),
        )

    def forward(self, x: torch.Tensor, token: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.to_params(token).chunk(2, dim=1)
        gamma = gamma[:, :, None, None]
        beta = beta[:, :, None, None]
        return x * (1.0 + gamma) + beta


class LowFrequencyBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.large_kernel = nn.Conv2d(channels, channels, 7, padding=3, groups=channels)
        self.mix = nn.Conv2d(channels, channels, 1)
        self.ffn = nn.Sequential(
            nn.Conv2d(channels, channels * 2, 1),
            nn.GELU(),
            nn.Conv2d(channels * 2, channels, 1),
        )
        self.norm = nn.GroupNorm(8 if channels >= 8 else 1, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.mix(self.large_kernel(x))
        y = self.ffn(y)
        return self.norm(x + y)


class DenseDegradationExpert(nn.Module):
    """Dense expert for dust, sand, haze, low-light, and color-cast."""

    def __init__(self, channels: int, token_dim: int, num_blocks: int = 4):
        super().__init__()
        self.token_pool = nn.Sequential(nn.LayerNorm(token_dim), nn.Linear(token_dim, token_dim), nn.GELU())
        self.film = FiLM(token_dim, channels)
        self.channel_gate = nn.Sequential(
            nn.Linear(token_dim, channels),
            nn.Sigmoid(),
        )
        self.blocks = nn.Sequential(*[LowFrequencyBlock(channels) for _ in range(num_blocks)])
        self.transmission_head = nn.Sequential(nn.Conv2d(channels, 1, 3, padding=1), nn.Sigmoid())
        self.illumination_head = nn.Sequential(nn.Conv2d(channels, 1, 3, padding=1), nn.Sigmoid())
        self.color_cast_head = nn.Conv2d(channels, 3, 3, padding=1)

    def forward(self, feature: torch.Tensor, dense_tokens: torch.Tensor) -> dict[str, torch.Tensor]:
        token = self.token_pool(dense_tokens.mean(dim=1))
        x = self.film(feature, token)
        gate = self.channel_gate(token)[:, :, None, None]
        x = x * gate
        dense_feature = self.blocks(x)
        return {
            "feature": dense_feature,
            "transmission": self.transmission_head(dense_feature),
            "illumination": self.illumination_head(dense_feature),
            "color_cast": self.color_cast_head(dense_feature),
        }
