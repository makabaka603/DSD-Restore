import torch
from torch import nn
from .backbone import (
    NAFNetRestorationDecoder,
    NAFNetSharedEncoder,
    RestorationDecoder,
    SharedEncoder,
)
from .experts import DenseDegradationExpert, SparseOcclusionExpert
from .fusion import SimpleDenseSparseFusion
from .prototype import CompositionalPrototypeBank
from .tokenizer import DegradationTokenizer


class DSDRestoreV1(nn.Module):
    """V1 model aligned with the project design.

    Pipeline:
    Input I -> Shared Encoder E -> Degradation Tokenizer T
    -> Compositional Prototype Bank -> Dense/Sparse Experts
    -> Simple Fusion -> Decoder D -> restored image.
    """

    supports_dsd_diagnostics = True

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_channels: int = 32,
        token_dim: int = 64,
        num_dense_tokens: int = 5,
        num_sparse_tokens: int = 4,
        backbone_type: str = "nafnet",
        prototype_temperature: float = 0.2,
        fusion_mode: str = "complementary",
        token_pooling: str = "mean",
        prototype_weighting: str = "softmax",
    ):
        super().__init__()
        backbone_type = backbone_type.lower()
        if backbone_type == "nafnet":
            self.encoder = NAFNetSharedEncoder(
                in_channels=in_channels,
                base_channels=base_channels,
            )
            decoder_cls = NAFNetRestorationDecoder
        elif backbone_type == "simple":
            self.encoder = SharedEncoder(in_channels=in_channels, base_channels=base_channels)
            decoder_cls = RestorationDecoder
        else:
            raise ValueError("backbone_type must be 'nafnet' or 'simple'")
        self.tokenizer = DegradationTokenizer(
            self.encoder.out_channels,
            token_dim=token_dim,
            num_dense_tokens=num_dense_tokens,
            num_sparse_tokens=num_sparse_tokens,
        )
        self.prototype_bank = CompositionalPrototypeBank(
            token_dim=token_dim,
            num_dense_prototypes=num_dense_tokens,
            num_sparse_prototypes=num_sparse_tokens,
            temperature=prototype_temperature,
            weighting_mode=prototype_weighting,
        )
        bottleneck_channels = self.encoder.out_channels[-1]
        self.dense_expert = DenseDegradationExpert(
            bottleneck_channels,
            token_dim,
            token_pooling=token_pooling,
        )
        self.sparse_expert = SparseOcclusionExpert(
            bottleneck_channels,
            token_dim,
            token_pooling=token_pooling,
        )
        self.fusion = SimpleDenseSparseFusion(
            bottleneck_channels,
            token_dim,
            gate_mode=fusion_mode,
        )
        self.decoder = decoder_cls(self.encoder.out_channels, out_channels=out_channels)

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.encoder(image)
        tokens = self.tokenizer(image, features)
        prototypes = self.prototype_bank(
            tokens["dense_tokens"],
            tokens["sparse_tokens"],
            tokens["dense_logits"],
            tokens["sparse_logits"],
        )
        dense = self.dense_expert(
            features[-1],
            tokens["dense_tokens"],
            tokens["dense_logits"],
            prototypes["dense_context"],
        )
        sparse = self.sparse_expert(
            features[-1],
            tokens["sparse_tokens"],
            tokens["sparse_logits"],
            prototypes["sparse_context"],
        )
        fused, dense_gate, sparse_gate = self.fusion(
            features[-1],
            dense["feature"],
            sparse["feature"],
            prototypes["dense_context"],
            prototypes["sparse_context"],
        )
        restored = self.decoder([features[0], features[1], features[2], fused])
        # Do not clamp during training: saturation would zero the gradient at
        # precisely the over/under-exposed pixels the model must correct.
        restored = image + torch.tanh(restored)
        return {
            "restored": restored,
            "dense_tokens": tokens["dense_tokens"],
            "sparse_tokens": tokens["sparse_tokens"],
            "dense_logits": tokens["dense_logits"],
            "sparse_logits": tokens["sparse_logits"],
            "dense_feature": dense["feature"],
            "sparse_feature": sparse["feature"],
            "sparse_mask": torch.nn.functional.interpolate(
                sparse["mask"], size=image.shape[-2:], mode="bilinear", align_corners=False
            ),
            # fusion_alpha is retained for old dashboards and checkpoints. In
            # V1.1 it is an alias of the independent dense gate.
            "fusion_alpha": dense_gate,
            "fusion_dense_gate": dense_gate,
            "fusion_sparse_gate": sparse_gate,
            "dense_prototype_weights": prototypes["dense_prototype_weights"],
            "sparse_prototype_weights": prototypes["sparse_prototype_weights"],
            "dense_prototype_activations": prototypes[
                "dense_prototype_activations"
            ],
            "sparse_prototype_activations": prototypes[
                "sparse_prototype_activations"
            ],
            "dense_prototypes": prototypes["dense_prototypes"],
            "sparse_prototypes": prototypes["sparse_prototypes"],
        }
