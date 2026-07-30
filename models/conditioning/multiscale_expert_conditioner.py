from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn


class DenseSparseScaleAdapter(nn.Module):
    """Low-cost dense/sparse conditioning with an identity-safe residual."""

    def __init__(
        self,
        channels: int,
        token_dim: int,
        reduction: int = 4,
    ):
        super().__init__()
        if reduction < 1:
            raise ValueError("multiscale reduction must be at least one")
        hidden_channels = max(channels // reduction, 8)
        self.dense_gate = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, channels),
            nn.Sigmoid(),
        )
        self.sparse_gate = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, channels),
            nn.Sigmoid(),
        )
        self.reduce = nn.Conv2d(channels, hidden_channels, 1)
        self.spatial = nn.Conv2d(
            hidden_channels,
            hidden_channels,
            3,
            padding=1,
            groups=hidden_channels,
        )
        self.expand = nn.Conv2d(hidden_channels, channels, 1)
        self.activation = nn.GELU()

        # Preserve the exact V1 function at initialization. The final
        # projection learns first and opens the residual path gradually.
        nn.init.zeros_(self.expand.weight)
        nn.init.zeros_(self.expand.bias)

    def forward(
        self,
        feature: torch.Tensor,
        dense_prompt: torch.Tensor,
        sparse_prompt: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        low_frequency = F.avg_pool2d(
            feature,
            kernel_size=7,
            stride=1,
            padding=3,
        )
        high_frequency = feature - low_frequency
        dense_gate = self.dense_gate(dense_prompt)[:, :, None, None]
        sparse_gate = self.sparse_gate(sparse_prompt)[:, :, None, None]
        conditioned = (
            dense_gate * low_frequency
            + sparse_gate * high_frequency
        )
        residual = self.expand(
            self.activation(
                self.spatial(
                    self.activation(self.reduce(conditioned))
                )
            )
        )
        return feature + residual, {
            "residual": residual,
            "dense_gate": dense_gate,
            "sparse_gate": sparse_gate,
        }


class MultiScaleExpertConditioner(nn.Module):
    """Apply identity-initialized dense/sparse adapters to selected scales."""

    def __init__(
        self,
        encoder_channels: Sequence[int],
        token_dim: int,
        levels: Sequence[int] = (1, 2, 3),
        reduction: int = 4,
    ):
        super().__init__()
        normalized_levels = tuple(int(level) for level in levels)
        if not normalized_levels:
            raise ValueError("multiscale levels cannot be empty")
        if len(set(normalized_levels)) != len(normalized_levels):
            raise ValueError("multiscale levels must be unique")
        invalid = [
            level
            for level in normalized_levels
            if level < 0 or level >= len(encoder_channels)
        ]
        if invalid:
            raise ValueError(
                f"Invalid multiscale levels {invalid} for "
                f"{len(encoder_channels)} encoder scales"
            )
        self.levels = normalized_levels
        self.adapters = nn.ModuleDict(
            {
                f"f{level + 1}": DenseSparseScaleAdapter(
                    channels=int(encoder_channels[level]),
                    token_dim=token_dim,
                    reduction=reduction,
                )
                for level in self.levels
            }
        )

    def forward(
        self,
        features: list[torch.Tensor],
        dense_prompt: torch.Tensor,
        sparse_prompt: torch.Tensor,
    ) -> tuple[list[torch.Tensor], dict[str, torch.Tensor]]:
        conditioned_features = list(features)
        diagnostics: dict[str, torch.Tensor] = {}
        for level in self.levels:
            name = f"f{level + 1}"
            conditioned, adapter_outputs = self.adapters[name](
                conditioned_features[level],
                dense_prompt,
                sparse_prompt,
            )
            conditioned_features[level] = conditioned
            diagnostics[f"multiscale_residual_{name}"] = adapter_outputs[
                "residual"
            ]
            diagnostics[f"multiscale_dense_gate_{name}"] = adapter_outputs[
                "dense_gate"
            ]
            diagnostics[f"multiscale_sparse_gate_{name}"] = adapter_outputs[
                "sparse_gate"
            ]
        return conditioned_features, diagnostics
