from __future__ import annotations

import torch


class PerceptualMetrics:
    """Reusable LPIPS and DISTS evaluators.

    The networks are constructed once because rebuilding them for every batch is
    both slow and can repeatedly load their pretrained weights.
    """

    def __init__(self, device: torch.device) -> None:
        try:
            import lpips
            from DISTS_pytorch import DISTS
        except ImportError as exc:
            raise ImportError(
                "LPIPS/DISTS metrics are enabled but their packages are missing. "
                "Run: python -m pip install lpips DISTS-pytorch"
            ) from exc

        self.lpips = lpips.LPIPS(net="alex").to(device).eval()
        self.dists = DISTS().to(device).eval()
        for metric in (self.lpips, self.dists):
            for parameter in metric.parameters():
                parameter.requires_grad_(False)

    @torch.no_grad()
    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
        # LPIPS expects [-1, 1]; DISTS expects ImageNet-style RGB in [0, 1].
        pred = pred.detach().float().clamp(0, 1)
        target = target.detach().float().clamp(0, 1)
        lpips_value = self.lpips(pred * 2 - 1, target * 2 - 1).mean()
        dists_value = self.dists(pred, target).mean()
        return {"lpips": float(lpips_value.item()), "dists": float(dists_value.item())}
