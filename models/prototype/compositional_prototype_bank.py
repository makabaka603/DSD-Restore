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
    ):
        super().__init__()
        if temperature <= 0:
            raise ValueError("prototype temperature must be greater than zero")
        self.temperature = temperature
        self.dense_prototypes = nn.Parameter(torch.empty(num_dense_prototypes, token_dim))
        self.sparse_prototypes = nn.Parameter(torch.empty(num_sparse_prototypes, token_dim))
        nn.init.trunc_normal_(self.dense_prototypes, std=0.02)
        nn.init.trunc_normal_(self.sparse_prototypes, std=0.02)

    def _compose(
        self,
        tokens: torch.Tensor,
        logits: torch.Tensor,
        prototypes: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if tokens.shape[1] != prototypes.shape[0]:
            raise ValueError(
                f"token/prototype count mismatch: {tokens.shape[1]} vs {prototypes.shape[0]}"
            )
        token_norm = F.normalize(tokens, dim=-1)
        prototype_norm = F.normalize(prototypes, dim=-1)
        similarity = (token_norm * prototype_norm.unsqueeze(0)).sum(dim=-1)
        presence = torch.sigmoid(logits)
        mixture = torch.softmax(similarity / self.temperature, dim=1)
        gated_weights = mixture * presence
        # Preserve the absolute presence magnitude in the context so an absent
        # degradation family approaches a zero prompt. A normalized copy is
        # returned for composition supervision and visualization.
        context = gated_weights @ prototypes
        normalized_weights = gated_weights / gated_weights.sum(
            dim=1, keepdim=True
        ).clamp_min(1e-6)
        return context, normalized_weights

    def forward(
        self,
        dense_tokens: torch.Tensor,
        sparse_tokens: torch.Tensor,
        dense_logits: torch.Tensor,
        sparse_logits: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        dense_context, dense_weights = self._compose(
            dense_tokens, dense_logits, self.dense_prototypes
        )
        sparse_context, sparse_weights = self._compose(
            sparse_tokens, sparse_logits, self.sparse_prototypes
        )
        return {
            "dense_context": dense_context,
            "sparse_context": sparse_context,
            "dense_prototype_weights": dense_weights,
            "sparse_prototype_weights": sparse_weights,
            "dense_prototypes": self.dense_prototypes,
            "sparse_prototypes": self.sparse_prototypes,
        }
