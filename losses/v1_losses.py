import torch
from torch import nn
import torch.nn.functional as F


def ssim_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    mu_x = F.avg_pool2d(pred, 3, 1, 1)
    mu_y = F.avg_pool2d(target, 3, 1, 1)
    sigma_x = F.avg_pool2d(pred * pred, 3, 1, 1) - mu_x * mu_x
    sigma_y = F.avg_pool2d(target * target, 3, 1, 1) - mu_y * mu_y
    sigma_xy = F.avg_pool2d(pred * target, 3, 1, 1) - mu_x * mu_y
    ssim = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
        (mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x + sigma_y + c2) + 1e-8
    )
    return 1.0 - ssim.mean()


def frequency_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_fft = torch.fft.rfft2(pred, norm="ortho")
    target_fft = torch.fft.rfft2(target, norm="ortho")
    return F.l1_loss(torch.abs(pred_fft), torch.abs(target_fft))


def color_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_mean = pred.mean(dim=(2, 3))
    target_mean = target.mean(dim=(2, 3))
    pred_std = pred.std(dim=(2, 3), unbiased=False)
    target_std = target.std(dim=(2, 3), unbiased=False)
    return F.l1_loss(pred_mean, target_mean) + 0.5 * F.l1_loss(pred_std, target_std)


def _prototype_diversity(prototypes: torch.Tensor) -> torch.Tensor:
    normalized = F.normalize(prototypes, dim=-1)
    similarity = normalized @ normalized.transpose(0, 1)
    identity = torch.eye(similarity.shape[0], device=similarity.device, dtype=similarity.dtype)
    return ((similarity - identity) ** 2).mean()


def _composition_loss(weights: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    active = labels.sum(dim=1) > 0
    if not bool(active.any()):
        return weights.new_zeros(())
    target = labels[active] / labels[active].sum(dim=1, keepdim=True).clamp_min(1e-6)
    return F.kl_div(weights[active].clamp_min(1e-6).log(), target, reduction="batchmean")


def _multilabel_prototype_loss(
    activations: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    # Probability-space BCE is kept in FP32 for the same AMP reason as the
    # sparse-mask loss below.
    with torch.amp.autocast(device_type=activations.device.type, enabled=False):
        probabilities = activations.float().clamp(1e-6, 1.0 - 1e-6)
        return F.binary_cross_entropy(probabilities, labels.float())


def prototype_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    mode: str = "composition",
) -> torch.Tensor:
    mode = mode.lower()
    if mode not in {"composition", "multilabel"}:
        raise ValueError("prototype loss mode must be 'composition' or 'multilabel'")
    diversity = _prototype_diversity(outputs["dense_prototypes"])
    diversity = diversity + _prototype_diversity(outputs["sparse_prototypes"])
    if mode == "composition":
        composition = _composition_loss(
            outputs["dense_prototype_weights"], batch["dense_label"]
        )
        composition = composition + _composition_loss(
            outputs["sparse_prototype_weights"], batch["sparse_label"]
        )
    else:
        composition = _multilabel_prototype_loss(
            outputs["dense_prototype_activations"], batch["dense_label"]
        )
        composition = composition + _multilabel_prototype_loss(
            outputs["sparse_prototype_activations"], batch["sparse_label"]
        )
    return diversity + 0.5 * composition


def sparse_mask_loss(
    mask: torch.Tensor,
    sparse_label: torch.Tensor,
    mode: str = "legacy",
    topk_fraction: float = 0.05,
    min_positive_coverage: float = 0.02,
) -> torch.Tensor:
    """Weakly supervise a sparse-degradation mask from image-level labels."""
    mode = mode.lower()
    if mode not in {"legacy", "topk"}:
        raise ValueError("sparse mask loss mode must be 'legacy' or 'topk'")
    if not 0.0 < topk_fraction <= 1.0:
        raise ValueError("topk_fraction must be in (0, 1]")
    if not 0.0 <= min_positive_coverage <= 1.0:
        raise ValueError("min_positive_coverage must be in [0, 1]")
    # CUDA autocast intentionally rejects probability-space BCE because its
    # backward pass can overflow in float16/bfloat16. Keep this small regularizer
    # in FP32 while the model and the remaining losses continue to use AMP.
    # Casting before clamping is also important: 1 - 1e-6 rounds back to 1 in
    # bfloat16, which would otherwise make BCE numerically unstable.
    with torch.amp.autocast(device_type=mask.device.type, enabled=False):
        mask_float = mask.float()
        has_sparse = sparse_label.float().amax(dim=1)
        mask_flat = mask_float.flatten(1)
        if mode == "legacy":
            presence = mask_flat.amax(dim=1)
        else:
            # Requiring the mean of the strongest region to be high prevents
            # the single-hot-pixel solution encouraged by a global maximum.
            topk_count = max(1, round(mask_flat.shape[1] * topk_fraction))
            presence = mask_flat.topk(topk_count, dim=1).values.mean(dim=1)
        presence = presence.clamp(1e-6, 1.0 - 1e-6)
        presence_loss = F.binary_cross_entropy(presence, has_sparse)
        mean_coverage = mask_flat.mean(dim=1)
        coverage_loss = ((1.0 - has_sparse) * mean_coverage).mean()
        if mode == "legacy":
            coverage_loss = coverage_loss + 0.05 * (
                has_sparse * mean_coverage
            ).mean()
        else:
            # A one-sided floor avoids collapsed positive masks without
            # prescribing a fixed rain/snow area for every image.
            coverage_floor = torch.relu(
                min_positive_coverage - mean_coverage
            ).square()
            coverage_loss = coverage_loss + (
                has_sparse * coverage_floor
            ).mean()
        tv_h = torch.abs(mask_float[:, :, 1:, :] - mask_float[:, :, :-1, :]).mean()
        tv_w = torch.abs(mask_float[:, :, :, 1:] - mask_float[:, :, :, :-1]).mean()
        return presence_loss + coverage_loss + 0.1 * (tv_h + tv_w)


class DSDRestoreV1Loss(nn.Module):
    def __init__(
        self,
        lambda_rec: float = 1.0,
        lambda_ssim: float = 0.2,
        lambda_freq: float = 0.05,
        lambda_color: float = 0.05,
        lambda_cls: float = 0.05,
        lambda_proto: float = 0.02,
        lambda_sparse: float = 0.01,
        classification_pos_weight: float | list[float] = 1.0,
        prototype_loss_mode: str = "composition",
        mask_loss_mode: str = "legacy",
        mask_topk_fraction: float = 0.05,
        mask_min_positive_coverage: float = 0.02,
        auxiliary_decay_start: float = 1.0,
        auxiliary_final_scale: float = 1.0,
    ):
        super().__init__()
        self.lambda_rec = lambda_rec
        self.lambda_ssim = lambda_ssim
        self.lambda_freq = lambda_freq
        self.lambda_color = lambda_color
        self.lambda_cls = lambda_cls
        self.lambda_proto = lambda_proto
        self.lambda_sparse = lambda_sparse
        if isinstance(classification_pos_weight, list):
            if not classification_pos_weight or any(
                float(value) <= 0 for value in classification_pos_weight
            ):
                raise ValueError("classification_pos_weight values must be positive")
            self.classification_pos_weight = [
                float(value) for value in classification_pos_weight
            ]
        else:
            if float(classification_pos_weight) <= 0:
                raise ValueError("classification_pos_weight must be positive")
            self.classification_pos_weight = float(classification_pos_weight)
        prototype_loss_mode = prototype_loss_mode.lower()
        if prototype_loss_mode not in {"composition", "multilabel"}:
            raise ValueError(
                "prototype_loss_mode must be 'composition' or 'multilabel'"
            )
        self.prototype_loss_mode = prototype_loss_mode
        self.mask_loss_mode = mask_loss_mode.lower()
        self.mask_topk_fraction = float(mask_topk_fraction)
        self.mask_min_positive_coverage = float(mask_min_positive_coverage)
        if not 0.0 <= auxiliary_decay_start <= 1.0:
            raise ValueError("auxiliary_decay_start must be in [0, 1]")
        if not 0.0 <= auxiliary_final_scale <= 1.0:
            raise ValueError("auxiliary_final_scale must be in [0, 1]")
        self.auxiliary_decay_start = float(auxiliary_decay_start)
        self.auxiliary_final_scale = float(auxiliary_final_scale)
        self.auxiliary_scale = 1.0

    def set_training_progress(self, progress: float) -> None:
        progress = min(max(float(progress), 0.0), 1.0)
        if progress <= self.auxiliary_decay_start:
            self.auxiliary_scale = 1.0
            return
        decay_span = max(1e-8, 1.0 - self.auxiliary_decay_start)
        fraction = (progress - self.auxiliary_decay_start) / decay_span
        self.auxiliary_scale = 1.0 + fraction * (
            self.auxiliary_final_scale - 1.0
        )

    def _pos_weight(
        self,
        logits: torch.Tensor,
    ) -> torch.Tensor:
        configured = self.classification_pos_weight
        if isinstance(configured, list):
            if len(configured) not in {1, logits.shape[1]}:
                raise ValueError(
                    "classification_pos_weight list must contain one value or "
                    f"{logits.shape[1]} values, got {len(configured)}"
                )
            return torch.tensor(
                configured,
                device=logits.device,
                dtype=logits.dtype,
            )
        return torch.tensor(
            configured,
            device=logits.device,
            dtype=logits.dtype,
        )

    @property
    def component_weights(self) -> dict[str, float]:
        return {
            "rec": self.lambda_rec,
            "ssim": self.lambda_ssim,
            "freq": self.lambda_freq,
            "color": self.lambda_color,
            "cls": self.lambda_cls * self.auxiliary_scale,
            "proto": self.lambda_proto * self.auxiliary_scale,
            "sparse": self.lambda_sparse * self.auxiliary_scale,
        }

    def weighted_components(
        self, losses: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        return {
            name: losses[name] * weight
            for name, weight in self.component_weights.items()
        }

    def forward(self, outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        pred = outputs["restored"]
        target = batch["gt"]
        rec = F.l1_loss(pred, target)
        ssim = ssim_loss(pred, target)
        freq = frequency_loss(pred, target)
        color = color_loss(pred, target)
        zero = pred.new_zeros(())
        if self.lambda_cls > 0:
            dense_label = batch["dense_label"]
            sparse_label = batch["sparse_label"]
            cls = F.binary_cross_entropy_with_logits(
                outputs["dense_logits"],
                dense_label,
                pos_weight=self._pos_weight(outputs["dense_logits"]),
            )
            cls = cls + F.binary_cross_entropy_with_logits(
                outputs["sparse_logits"],
                sparse_label,
                pos_weight=self._pos_weight(outputs["sparse_logits"]),
            )
        else:
            cls = zero
        proto = (
            prototype_loss(outputs, batch, mode=self.prototype_loss_mode)
            if self.lambda_proto > 0
            else zero
        )
        sparse = (
            sparse_mask_loss(
                outputs["sparse_mask"],
                batch["sparse_label"],
                mode=self.mask_loss_mode,
                topk_fraction=self.mask_topk_fraction,
                min_positive_coverage=self.mask_min_positive_coverage,
            )
            if self.lambda_sparse > 0
            else zero
        )
        weights = self.component_weights
        total = (
            weights["rec"] * rec
            + weights["ssim"] * ssim
            + weights["freq"] * freq
            + weights["color"] * color
            + weights["cls"] * cls
            + weights["proto"] * proto
            + weights["sparse"] * sparse
        )
        return {
            "total": total,
            "rec": rec.detach(),
            "ssim": ssim.detach(),
            "freq": freq.detach(),
            "color": color.detach(),
            "cls": cls.detach(),
            "proto": proto.detach(),
            "sparse": sparse.detach(),
        }
