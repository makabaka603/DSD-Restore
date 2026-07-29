import torch
from torch import nn


def pool_tokens(
    tokens: torch.Tensor,
    logits: torch.Tensor,
    mode: str,
) -> torch.Tensor:
    if mode == "mean":
        return tokens.mean(dim=1)
    presence = torch.sigmoid(logits).unsqueeze(-1)
    # clamp_min(1) preserves absolute absence: if every class is unlikely,
    # the pooled prompt approaches zero instead of amplifying numerical noise.
    return (tokens * presence).sum(dim=1) / presence.sum(dim=1).clamp_min(1.0)


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
        self.token_pool = nn.Sequential(nn.LayerNorm(token_dim), nn.Linear(token_dim, token_dim), nn.GELU())
        self.film = FiLM(token_dim, channels)
        self.channel_gate = nn.Sequential(
            nn.Linear(token_dim, channels),
            nn.Sigmoid(),
        )
        self.blocks = nn.Sequential(*[LowFrequencyBlock(channels) for _ in range(num_blocks)])

    def forward(
        self,
        feature: torch.Tensor,
        dense_tokens: torch.Tensor,
        dense_logits: torch.Tensor,
        prototype_context: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        token = self.token_pool(
            pool_tokens(dense_tokens, dense_logits, self.token_pooling)
            + prototype_context
        )
        x = self.film(feature, token)
        gate = self.channel_gate(token)[:, :, None, None]
        x = x * gate
        dense_feature = self.blocks(x)
        return {"feature": dense_feature}
