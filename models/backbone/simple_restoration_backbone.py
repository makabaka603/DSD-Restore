import torch
from torch import nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
        )
        self.skip = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        self.norm = nn.GroupNorm(8 if out_channels >= 8 else 1, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.block(x) + self.skip(x))


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.down = nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1)
        self.conv = ConvBlock(out_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.down(x))


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, out_channels, 1)
        self.conv = ConvBlock(out_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = self.proj(x)
        return self.conv(torch.cat([x, skip], dim=1))


class SharedEncoder(nn.Module):
    """V1 shared feature encoder E. It provides multi-scale features F1...F4."""

    def __init__(self, in_channels: int = 3, base_channels: int = 32):
        super().__init__()
        c = base_channels
        self.stem = ConvBlock(in_channels, c)
        self.stage1 = ConvBlock(c, c)
        self.stage2 = DownBlock(c, c * 2)
        self.stage3 = DownBlock(c * 2, c * 4)
        self.stage4 = DownBlock(c * 4, c * 8)
        self.out_channels = [c, c * 2, c * 4, c * 8]

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        x = self.stem(x)
        f1 = self.stage1(x)
        f2 = self.stage2(f1)
        f3 = self.stage3(f2)
        f4 = self.stage4(f3)
        return [f1, f2, f3, f4]


class RestorationDecoder(nn.Module):
    """V1 restoration decoder D with multi-scale reconstruction."""

    def __init__(self, encoder_channels: list[int], out_channels: int = 3):
        super().__init__()
        c1, c2, c3, c4 = encoder_channels
        self.up3 = UpBlock(c4, c3, c3)
        self.up2 = UpBlock(c3, c2, c2)
        self.up1 = UpBlock(c2, c1, c1)
        self.refine = ConvBlock(c1, c1)
        self.out = nn.Conv2d(c1, out_channels, 3, padding=1)

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        f1, f2, f3, f4 = features
        x = self.up3(f4, f3)
        x = self.up2(x, f2)
        x = self.up1(x, f1)
        x = self.refine(x)
        return self.out(x)
