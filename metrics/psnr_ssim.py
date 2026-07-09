import math

import torch
import torch.nn.functional as F


def batch_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    mse = F.mse_loss(pred.clamp(0, 1), target.clamp(0, 1), reduction="none")
    mse = mse.flatten(1).mean(dim=1).clamp_min(1e-10)
    return float((10.0 * torch.log10(1.0 / mse)).mean().item())


def batch_ssim(pred: torch.Tensor, target: torch.Tensor) -> float:
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    pred = pred.clamp(0, 1)
    target = target.clamp(0, 1)
    mu_x = F.avg_pool2d(pred, 3, 1, 1)
    mu_y = F.avg_pool2d(target, 3, 1, 1)
    sigma_x = F.avg_pool2d(pred * pred, 3, 1, 1) - mu_x * mu_x
    sigma_y = F.avg_pool2d(target * target, 3, 1, 1) - mu_y * mu_y
    sigma_xy = F.avg_pool2d(pred * target, 3, 1, 1) - mu_x * mu_y
    ssim = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
        (mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x + sigma_y + c2) + 1e-8
    )
    value = float(ssim.mean().item())
    if math.isnan(value):
        return 0.0
    return value
