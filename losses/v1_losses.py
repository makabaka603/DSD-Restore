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


def prototype_loss(outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> torch.Tensor:
    diversity = _prototype_diversity(outputs["dense_prototypes"])
    diversity = diversity + _prototype_diversity(outputs["sparse_prototypes"])
    composition = _composition_loss(outputs["dense_prototype_weights"], batch["dense_label"])
    composition = composition + _composition_loss(
        outputs["sparse_prototype_weights"], batch["sparse_label"]
    )
    return diversity + 0.5 * composition


def sparse_mask_loss(mask: torch.Tensor, sparse_label: torch.Tensor) -> torch.Tensor:
    """Encourage sparse, non-empty masks only when an occlusion label is present."""
    # CUDA autocast intentionally rejects probability-space BCE because its
    # backward pass can overflow in float16/bfloat16. Keep this small regularizer
    # in FP32 while the model and the remaining losses continue to use AMP.
    # Casting before clamping is also important: 1 - 1e-6 rounds back to 1 in
    # bfloat16, which would otherwise make BCE numerically unstable.
    with torch.amp.autocast(device_type=mask.device.type, enabled=False):
        mask_float = mask.float()
        has_sparse = sparse_label.float().amax(dim=1)
        mask_flat = mask_float.flatten(1)
        presence = mask_flat.amax(dim=1).clamp(1e-6, 1.0 - 1e-6)
        presence_loss = F.binary_cross_entropy(presence, has_sparse)
        mean_coverage = mask_flat.mean(dim=1)
        coverage_loss = ((1.0 - has_sparse) * mean_coverage).mean()
        coverage_loss = coverage_loss + 0.05 * (has_sparse * mean_coverage).mean()
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
    ):
        super().__init__()
        self.lambda_rec = lambda_rec
        self.lambda_ssim = lambda_ssim
        self.lambda_freq = lambda_freq
        self.lambda_color = lambda_color
        self.lambda_cls = lambda_cls
        self.lambda_proto = lambda_proto
        self.lambda_sparse = lambda_sparse

    @property
    def component_weights(self) -> dict[str, float]:
        return {
            "rec": self.lambda_rec,
            "ssim": self.lambda_ssim,
            "freq": self.lambda_freq,
            "color": self.lambda_color,
            "cls": self.lambda_cls,
            "proto": self.lambda_proto,
            "sparse": self.lambda_sparse,
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
        dense_label = batch["dense_label"]
        sparse_label = batch["sparse_label"]
        rec = F.l1_loss(pred, target)
        ssim = ssim_loss(pred, target)
        freq = frequency_loss(pred, target)
        color = color_loss(pred, target)
        cls = F.binary_cross_entropy_with_logits(outputs["dense_logits"], dense_label)
        cls = cls + F.binary_cross_entropy_with_logits(outputs["sparse_logits"], sparse_label)
        proto = prototype_loss(outputs, batch)
        sparse = sparse_mask_loss(outputs["sparse_mask"], sparse_label)
        total = (
            self.lambda_rec * rec
            + self.lambda_ssim * ssim
            + self.lambda_freq * freq
            + self.lambda_color * color
            + self.lambda_cls * cls
            + self.lambda_proto * proto
            + self.lambda_sparse * sparse
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
