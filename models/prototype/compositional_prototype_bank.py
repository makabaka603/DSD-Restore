import torch
from torch import nn
import torch.nn.functional as F


class CompositionalPrototypeBank(nn.Module):
    """Dense/sparse prototype banks for continuous degradation composition."""

    def __init__(
        self,
        token_dim: int,
        num_dense_prototypes: int = 5,
        num_sparse_prototypes: int = 4,
        temperature: float = 0.2,
        weighting_mode: str = "softmax",
    ):
        super().__init__()
        if temperature <= 0:
            raise ValueError("prototype temperature must be greater than zero")
        weighting_mode = weighting_mode.lower()
        if weighting_mode not in {"softmax", "sigmoid"}:
            raise ValueError("weighting_mode must be 'softmax' or 'sigmoid'")
        self.temperature = temperature
        self.weighting_mode = weighting_mode
        self.dense_prototypes = nn.Parameter(torch.empty(num_dense_prototypes, token_dim))
        self.sparse_prototypes = nn.Parameter(torch.empty(num_sparse_prototypes, token_dim))
        nn.init.trunc_normal_(self.dense_prototypes, std=0.02)
        nn.init.trunc_normal_(self.sparse_prototypes, std=0.02)

    def _compose(
        self,
        tokens: torch.Tensor,
        logits: torch.Tensor,
        prototypes: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if tokens.shape[1] != prototypes.shape[0]:
            raise ValueError(
                f"token/prototype count mismatch: {tokens.shape[1]} vs {prototypes.shape[0]}"
            )
        token_norm = F.normalize(tokens, dim=-1)
        prototype_norm = F.normalize(prototypes, dim=-1)
        similarity = (token_norm * prototype_norm.unsqueeze(0)).sum(dim=-1)
        presence = torch.sigmoid(logits)
        if self.weighting_mode == "softmax":
            mixture = torch.softmax(similarity / self.temperature, dim=1)
            activations = mixture * presence
            # Preserve the exact V1 behavior for old configs and checkpoints.
            context = activations @ prototypes
        else:
            # Each prototype can activate independently. Dividing only when
            # total activation exceeds one keeps an absent family near zero
            # while preventing multi-label contexts from growing unbounded.
            similarity_gate = torch.sigmoid(similarity / self.temperature)
            activations = similarity_gate * presence
            context = (activations @ prototypes) / activations.sum(
                dim=1, keepdim=True
            ).clamp_min(1.0)
        normalized_weights = activations / activations.sum(
            dim=1, keepdim=True
        ).clamp_min(1e-6)
        return context, normalized_weights, activations

    def forward(
        self,
        dense_tokens: torch.Tensor,
        sparse_tokens: torch.Tensor,
        dense_logits: torch.Tensor,
        sparse_logits: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        dense_context, dense_weights, dense_activations = self._compose(
            dense_tokens, dense_logits, self.dense_prototypes
        )
        sparse_context, sparse_weights, sparse_activations = self._compose(
            sparse_tokens, sparse_logits, self.sparse_prototypes
        )
        return {
            "dense_context": dense_context,
            "sparse_context": sparse_context,
            "dense_prototype_weights": dense_weights,
            "sparse_prototype_weights": sparse_weights,
            "dense_prototype_activations": dense_activations,
            "sparse_prototype_activations": sparse_activations,
            "dense_prototypes": self.dense_prototypes,
            "sparse_prototypes": self.sparse_prototypes,
        }
