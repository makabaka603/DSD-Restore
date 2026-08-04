import math

import torch
from torch import nn


class FactorSpatialRouter(nn.Module):
    """Route dense/sparse expert features with factor-specific spatial maps.

    Each degradation token supplies a key and a channel value. A lightweight
    projection of the shared bottleneck supplies the spatial queries. Their
    correlation produces one map per degradation factor instead of one global
    dense/sparse decision for the whole image.

    The two channel-wise modulation scales start at zero. Consequently this
    module is an exact identity when added to a V1 Stage 2 checkpoint, while
    still receiving gradients on the first optimization step.
    """

    def __init__(
        self,
        channels: int,
        token_dim: int,
        router_dim: int = 32,
        temperature: float = 1.0,
    ):
        super().__init__()
        if router_dim < 1:
            raise ValueError("router_dim must be at least 1")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.router_dim = int(router_dim)
        self.temperature = float(temperature)
        self.spatial_norm = nn.GroupNorm(1, channels)
        self.token_norm = nn.LayerNorm(token_dim)
        self.query = nn.Conv2d(channels, self.router_dim, 1)
        self.key = nn.Linear(token_dim, self.router_dim)
        self.value = nn.Linear(token_dim, channels)
        self.dense_scale = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.sparse_scale = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def _route_family(
        self,
        query: torch.Tensor,
        feature: torch.Tensor,
        tokens: torch.Tensor,
        presence_logits: torch.Tensor,
        scale: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        normalized_tokens = self.token_norm(tokens)
        keys = self.key(normalized_tokens)
        values = torch.tanh(self.value(normalized_tokens))
        similarity = torch.einsum("brhw,bkr->bkhw", query, keys)
        similarity = similarity / math.sqrt(self.router_dim)
        factor_gates = torch.sigmoid(
            similarity / self.temperature
            + presence_logits[:, :, None, None]
        )
        presence = torch.sigmoid(presence_logits)[:, :, None, None]
        weights = factor_gates * presence
        denominator = weights.sum(dim=1, keepdim=True).clamp_min(1.0)
        spatial_context = torch.einsum(
            "bkhw,bkc->bchw", weights, values
        ) / denominator
        modulation = torch.tanh(spatial_context)
        # Bounded channel scales keep multiplicative routing in [0, 2] and
        # prevent a short screen from destabilizing the inherited experts.
        routed = feature * (1.0 + torch.tanh(scale) * modulation)
        return routed, factor_gates, modulation

    def forward(
        self,
        shared: torch.Tensor,
        dense_feature: torch.Tensor,
        sparse_feature: torch.Tensor,
        dense_tokens: torch.Tensor,
        sparse_tokens: torch.Tensor,
        dense_logits: torch.Tensor,
        sparse_logits: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        query = self.query(self.spatial_norm(shared))
        dense_routed, dense_gates, dense_modulation = self._route_family(
            query,
            dense_feature,
            dense_tokens,
            dense_logits,
            self.dense_scale,
        )
        sparse_routed, sparse_gates, sparse_modulation = self._route_family(
            query,
            sparse_feature,
            sparse_tokens,
            sparse_logits,
            self.sparse_scale,
        )
        return {
            "dense_feature": dense_routed,
            "sparse_feature": sparse_routed,
            "dense_factor_gates": dense_gates,
            "sparse_factor_gates": sparse_gates,
            "dense_modulation": dense_modulation,
            "sparse_modulation": sparse_modulation,
        }
