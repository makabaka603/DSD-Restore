import torch
from torch import nn
from .backbone import (
    NAFNetRestorationDecoder,
    NAFNetSharedEncoder,
    NAFStage,
    RestorationDecoder,
    SharedEncoder,
)
from .conditioning import MultiScaleExpertConditioner
from .experts import DenseDegradationExpert, SparseOcclusionExpert
from .experts.dense_expert import pool_tokens
from .fusion import SimpleDenseSparseFusion
from .prototype import CompositionalPrototypeBank
from .routing import FactorSpatialRouter
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
        multiscale_conditioning: bool = False,
        multiscale_levels: tuple[int, ...] = (1, 2, 3),
        multiscale_reduction: int = 4,
        dense_expert_blocks: int = 4,
        sparse_expert_blocks: int = 4,
        shared_bottleneck_blocks: int = 0,
        factor_spatial_routing: bool = False,
        factor_router_dim: int = 32,
        factor_router_temperature: float = 1.0,
        factor_pairwise_interaction: bool = False,
    ):
        super().__init__()
        if dense_expert_blocks < 1 or sparse_expert_blocks < 1:
            raise ValueError("Expert block counts must be at least 1")
        if shared_bottleneck_blocks < 0:
            raise ValueError("shared_bottleneck_blocks cannot be negative")
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
        bottleneck_channels = self.encoder.out_channels[-1]
        self.shared_bottleneck = (
            NAFStage(bottleneck_channels, shared_bottleneck_blocks)
            if shared_bottleneck_blocks > 0
            else nn.Identity()
        )
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
        self.token_pooling_mode = token_pooling
        self.multiscale_conditioner = (
            MultiScaleExpertConditioner(
                encoder_channels=self.encoder.out_channels,
                token_dim=token_dim,
                levels=multiscale_levels,
                reduction=multiscale_reduction,
            )
            if multiscale_conditioning
            else None
        )
        self.dense_expert = DenseDegradationExpert(
            bottleneck_channels,
            token_dim,
            num_blocks=dense_expert_blocks,
            token_pooling=token_pooling,
        )
        self.sparse_expert = SparseOcclusionExpert(
            bottleneck_channels,
            token_dim,
            num_blocks=sparse_expert_blocks,
            token_pooling=token_pooling,
        )
        self.factor_router = (
            FactorSpatialRouter(
                bottleneck_channels,
                token_dim,
                router_dim=factor_router_dim,
                temperature=factor_router_temperature,
                pairwise_interaction=factor_pairwise_interaction,
            )
            if factor_spatial_routing
            else None
        )
        self.fusion = SimpleDenseSparseFusion(
            bottleneck_channels,
            token_dim,
            gate_mode=fusion_mode,
        )
        self.decoder = decoder_cls(self.encoder.out_channels, out_channels=out_channels)

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.encoder(image)
        # D1 reallocates capacity from task-specific experts to this shared path.
        # NAFBlock beta/gamma parameters start at zero, so newly added blocks are
        # exact identity mappings when initialized from a legacy Stage 2 model.
        features[-1] = self.shared_bottleneck(features[-1])
        tokens = self.tokenizer(image, features)
        prototypes = self.prototype_bank(
            tokens["dense_tokens"],
            tokens["sparse_tokens"],
            tokens["dense_logits"],
            tokens["sparse_logits"],
        )
        multiscale_outputs: dict[str, torch.Tensor] = {}
        if self.multiscale_conditioner is not None:
            dense_prompt = (
                pool_tokens(
                    tokens["dense_tokens"],
                    tokens["dense_logits"],
                    self.token_pooling_mode,
                )
                + prototypes["dense_context"]
            )
            sparse_prompt = (
                pool_tokens(
                    tokens["sparse_tokens"],
                    tokens["sparse_logits"],
                    self.token_pooling_mode,
                )
                + prototypes["sparse_context"]
            )
            features, multiscale_outputs = self.multiscale_conditioner(
                features,
                dense_prompt,
                sparse_prompt,
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
        factor_routing_outputs: dict[str, torch.Tensor] = {}
        if self.factor_router is not None:
            factor_routing_outputs = self.factor_router(
                features[-1],
                dense["feature"],
                sparse["feature"],
                tokens["dense_tokens"],
                tokens["sparse_tokens"],
                tokens["dense_logits"],
                tokens["sparse_logits"],
            )
            dense["feature"] = factor_routing_outputs["dense_feature"]
            sparse["feature"] = factor_routing_outputs["sparse_feature"]
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
        outputs = {
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
        if factor_routing_outputs:
            routing_diagnostics = {
                "factor_dense_spatial_gates": factor_routing_outputs[
                    "dense_factor_gates"
                ],
                "factor_sparse_spatial_gates": factor_routing_outputs[
                    "sparse_factor_gates"
                ],
                "factor_dense_modulation": factor_routing_outputs[
                    "dense_modulation"
                ],
                "factor_sparse_modulation": factor_routing_outputs[
                    "sparse_modulation"
                ],
            }
            if "dense_pair_modulation" in factor_routing_outputs:
                routing_diagnostics.update(
                    {
                        "factor_dense_pair_modulation": factor_routing_outputs[
                            "dense_pair_modulation"
                        ],
                        "factor_sparse_pair_modulation": factor_routing_outputs[
                            "sparse_pair_modulation"
                        ],
                        "factor_dense_pair_scale": factor_routing_outputs[
                            "dense_pair_scale"
                        ],
                        "factor_sparse_pair_scale": factor_routing_outputs[
                            "sparse_pair_scale"
                        ],
                    }
                )
            outputs.update(routing_diagnostics)
        outputs.update(multiscale_outputs)
        return outputs


class DSDRestoreV2(DSDRestoreV1):
    """V2 with degradation-factor spatial routing at the bottleneck."""

    def __init__(
        self,
        factor_router_dim: int = 32,
        factor_router_temperature: float = 1.0,
        factor_pairwise_interaction: bool = False,
        **kwargs,
    ):
        kwargs.pop("factor_spatial_routing", None)
        super().__init__(
            factor_spatial_routing=True,
            factor_router_dim=factor_router_dim,
            factor_router_temperature=factor_router_temperature,
            factor_pairwise_interaction=factor_pairwise_interaction,
            **kwargs,
        )
