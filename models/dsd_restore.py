import torch
from torch import nn
import torch.nn.functional as F

from .backbone import SharedEncoder, RestorationDecoder
from .experts import DenseDegradationExpert, SparseOcclusionExpert
from .fusion import SimpleDenseSparseFusion
from .tokenizer import DegradationTokenizer


class DSDRestoreV1(nn.Module):
    """V1 model aligned with the project design.

    Pipeline:
    Input I -> Shared Encoder E -> Degradation Tokenizer T
    -> Dense/Sparse Experts -> Simple Fusion -> Decoder D -> restored image.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_channels: int = 32,
        token_dim: int = 64,
        num_dense_tokens: int = 5,
        num_sparse_tokens: int = 4,
    ):
        super().__init__()
        self.encoder = SharedEncoder(in_channels=in_channels, base_channels=base_channels)
        self.tokenizer = DegradationTokenizer(
            self.encoder.out_channels,
            token_dim=token_dim,
            num_dense_tokens=num_dense_tokens,
            num_sparse_tokens=num_sparse_tokens,
        )
        bottleneck_channels = self.encoder.out_channels[-1]
        self.dense_expert = DenseDegradationExpert(bottleneck_channels, token_dim)
        self.sparse_expert = SparseOcclusionExpert(bottleneck_channels, token_dim)
        self.fusion = SimpleDenseSparseFusion(bottleneck_channels, token_dim)
        self.decoder = RestorationDecoder(self.encoder.out_channels, out_channels=out_channels)

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.encoder(image)
        tokens = self.tokenizer(image, features)
        dense = self.dense_expert(features[-1], tokens["dense_tokens"])
        sparse = self.sparse_expert(features[-1], tokens["sparse_tokens"])
        fused, alpha = self.fusion(
            features[-1],
            dense["feature"],
            sparse["feature"],
            tokens["dense_tokens"],
            tokens["sparse_tokens"],
        )
        restored = self.decoder([features[0], features[1], features[2], fused])
        restored = torch.clamp(image + torch.tanh(restored), 0.0, 1.0)
        return {
            "restored": restored,
            "dense_tokens": tokens["dense_tokens"],
            "sparse_tokens": tokens["sparse_tokens"],
            "dense_logits": tokens["dense_logits"],
            "sparse_logits": tokens["sparse_logits"],
            "dense_feature": dense["feature"],
            "sparse_feature": sparse["feature"],
            "transmission": F.interpolate(dense["transmission"], size=image.shape[-2:], mode="bilinear", align_corners=False),
            "illumination": F.interpolate(dense["illumination"], size=image.shape[-2:], mode="bilinear", align_corners=False),
            "color_cast": F.interpolate(dense["color_cast"], size=image.shape[-2:], mode="bilinear", align_corners=False),
            "sparse_mask": F.interpolate(sparse["mask"], size=image.shape[-2:], mode="bilinear", align_corners=False),
            "fusion_alpha": alpha,
        }
