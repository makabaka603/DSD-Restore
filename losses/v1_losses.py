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


class DSDRestoreV1Loss(nn.Module):
    def __init__(
        self,
        lambda_rec: float = 1.0,
        lambda_ssim: float = 0.2,
        lambda_freq: float = 0.05,
        lambda_color: float = 0.05,
        lambda_cls: float = 0.05,
    ):
        super().__init__()
        self.lambda_rec = lambda_rec
        self.lambda_ssim = lambda_ssim
        self.lambda_freq = lambda_freq
        self.lambda_color = lambda_color
        self.lambda_cls = lambda_cls

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
        total = (
            self.lambda_rec * rec
            + self.lambda_ssim * ssim
            + self.lambda_freq * freq
            + self.lambda_color * color
            + self.lambda_cls * cls
        )
        return {
            "total": total,
            "rec": rec.detach(),
            "ssim": ssim.detach(),
            "freq": freq.detach(),
            "color": color.detach(),
            "cls": cls.detach(),
        }
