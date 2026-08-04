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
        pairwise_interaction: bool = False,
    ):
        super().__init__()
        if router_dim < 1:
            raise ValueError("router_dim must be at least 1")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.router_dim = int(router_dim)
        self.temperature = float(temperature)
        self.pairwise_interaction = bool(pairwise_interaction)
        self.spatial_norm = nn.GroupNorm(1, channels)
        self.token_norm = nn.LayerNorm(token_dim)
        self.query = nn.Conv2d(channels, self.router_dim, 1)
        self.key = nn.Linear(token_dim, self.router_dim)
        self.value = nn.Linear(token_dim, channels)
        self.dense_scale = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.sparse_scale = nn.Parameter(torch.zeros(1, channels, 1, 1))
        if self.pairwise_interaction:
            # Separate zero-initialized residual scales make V2.1 exactly
            # equivalent to V2 at initialization while allowing pairwise
            # factor interactions to receive gradients on the first step.
            self.dense_pair_scale = nn.Parameter(
                torch.zeros(1, channels, 1, 1)
            )
            self.sparse_pair_scale = nn.Parameter(
                torch.zeros(1, channels, 1, 1)
            )
        else:
            self.register_parameter("dense_pair_scale", None)
            self.register_parameter("sparse_pair_scale", None)

    @staticmethod
    def _pairwise_modulation(
        weights: torch.Tensor,
        values: torch.Tensor,
    ) -> torch.Tensor:
        """Return normalized second-order interactions between factor values."""
        batch, factors, height, width = weights.shape
        channels = values.shape[-1]
        numerator = weights.new_zeros((batch, channels, height, width))
        denominator = weights.new_zeros((batch, 1, height, width))
        for left in range(factors):
            for right in range(left + 1, factors):
                pair_weight = weights[:, left] * weights[:, right]
                pair_value = values[:, left] * values[:, right]
                numerator = numerator + (
                    pair_weight[:, None] * pair_value[:, :, None, None]
                )
                denominator = denominator + pair_weight[:, None]
        pair_context = numerator / denominator.clamp_min(1.0)
        return torch.tanh(pair_context)

    def _route_family(
        self,
        query: torch.Tensor,
        feature: torch.Tensor,
        tokens: torch.Tensor,
        presence_logits: torch.Tensor,
        scale: torch.Tensor,
        pair_scale: torch.Tensor | None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
    ]:
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
        base_gain = torch.tanh(scale) * modulation
        pair_modulation = None
        if pair_scale is not None:
            pair_modulation = self._pairwise_modulation(weights, values)
            # Use the unused portion of the base gain as headroom. This keeps
            # the combined multiplicative gain in [-1, 1], preserves V2 when
            # pair_scale is zero, and avoids destabilizing inherited experts.
            pair_headroom = 1.0 - base_gain.abs()
            base_gain = base_gain + (
                pair_headroom
                * torch.tanh(pair_scale)
                * pair_modulation
            )
        routed = feature * (1.0 + base_gain)
        return routed, factor_gates, modulation, pair_modulation

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
        dense_routed, dense_gates, dense_modulation, dense_pair_modulation = (
            self._route_family(
                query,
                dense_feature,
                dense_tokens,
                dense_logits,
                self.dense_scale,
                self.dense_pair_scale,
            )
        )
        sparse_routed, sparse_gates, sparse_modulation, sparse_pair_modulation = (
            self._route_family(
                query,
                sparse_feature,
                sparse_tokens,
                sparse_logits,
                self.sparse_scale,
                self.sparse_pair_scale,
            )
        )
        outputs = {
            "dense_feature": dense_routed,
            "sparse_feature": sparse_routed,
            "dense_factor_gates": dense_gates,
            "sparse_factor_gates": sparse_gates,
            "dense_modulation": dense_modulation,
            "sparse_modulation": sparse_modulation,
        }
        if (
            dense_pair_modulation is not None
            and sparse_pair_modulation is not None
        ):
            outputs.update(
                {
                    "dense_pair_modulation": dense_pair_modulation,
                    "sparse_pair_modulation": sparse_pair_modulation,
                    "dense_pair_scale": torch.tanh(self.dense_pair_scale),
                    "sparse_pair_scale": torch.tanh(self.sparse_pair_scale),
                }
            )
        return outputs
