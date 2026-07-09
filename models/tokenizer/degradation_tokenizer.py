import torch
from torch import nn
import torch.nn.functional as F


class DegradationTokenizer(nn.Module):
    """V1 degradation tokenizer T.

    It combines global pooled multi-scale features and shallow image statistics,
    then emits dense and sparse degradation tokens plus auxiliary logits.
    """

    dense_names = ("dust", "sand", "haze", "lowlight", "colorcast")
    sparse_names = ("rain", "raindrop", "snow", "occlusion")

    def __init__(
        self,
        encoder_channels: list[int],
        token_dim: int = 64,
        num_dense_tokens: int = 5,
        num_sparse_tokens: int = 4,
    ):
        super().__init__()
        self.num_dense_tokens = num_dense_tokens
        self.num_sparse_tokens = num_sparse_tokens
        pooled_dim = sum(encoder_channels)
        stats_dim = 3 + 3 + 1 + 1
        hidden = max(token_dim * 4, 128)
        self.mlp = nn.Sequential(
            nn.Linear(pooled_dim + stats_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.dense_proj = nn.Linear(hidden, num_dense_tokens * token_dim)
        self.sparse_proj = nn.Linear(hidden, num_sparse_tokens * token_dim)
        self.dense_cls = nn.Linear(token_dim, 1)
        self.sparse_cls = nn.Linear(token_dim, 1)

    def forward(self, image: torch.Tensor, features: list[torch.Tensor]) -> dict[str, torch.Tensor]:
        pooled = [F.adaptive_avg_pool2d(feat, 1).flatten(1) for feat in features]
        stats = self._image_statistics(image)
        hidden = self.mlp(torch.cat([*pooled, stats], dim=1))
        b = image.shape[0]
        dense_tokens = self.dense_proj(hidden).view(b, self.num_dense_tokens, -1)
        sparse_tokens = self.sparse_proj(hidden).view(b, self.num_sparse_tokens, -1)
        dense_logits = self.dense_cls(dense_tokens).squeeze(-1)
        sparse_logits = self.sparse_cls(sparse_tokens).squeeze(-1)
        return {
            "dense_tokens": dense_tokens,
            "sparse_tokens": sparse_tokens,
            "dense_logits": dense_logits,
            "sparse_logits": sparse_logits,
            "stats": stats,
        }

    @staticmethod
    def _image_statistics(image: torch.Tensor) -> torch.Tensor:
        mean = image.mean(dim=(2, 3))
        std = image.std(dim=(2, 3), unbiased=False)
        gray = image.mean(dim=1, keepdim=True)
        edge_h = torch.abs(gray[:, :, 1:, :] - gray[:, :, :-1, :]).mean(dim=(1, 2, 3), keepdim=False).unsqueeze(1)
        edge_w = torch.abs(gray[:, :, :, 1:] - gray[:, :, :, :-1]).mean(dim=(1, 2, 3), keepdim=False).unsqueeze(1)
        edge = 0.5 * (edge_h + edge_w)
        low = F.avg_pool2d(gray, kernel_size=7, stride=1, padding=3)
        high_energy = torch.abs(gray - low).mean(dim=(1, 2, 3), keepdim=False).unsqueeze(1)
        return torch.cat([mean, std, edge, high_energy], dim=1)
