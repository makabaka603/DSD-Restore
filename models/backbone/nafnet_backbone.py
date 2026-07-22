import torch
from torch import nn
import torch.nn.functional as F


class LayerNorm2d(nn.Module):
    """Channel-wise LayerNorm used by NAFNet blocks."""

    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class SimpleGate(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    """Compact NAFNet block suitable for the single-RTX-5090 V1 model."""

    def __init__(self, channels: int, expansion: int = 2, ffn_expansion: int = 2):
        super().__init__()
        depthwise_channels = channels * expansion
        ffn_channels = channels * ffn_expansion
        self.norm1 = LayerNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, depthwise_channels, 1)
        self.depthwise = nn.Conv2d(
            depthwise_channels,
            depthwise_channels,
            3,
            padding=1,
            groups=depthwise_channels,
        )
        self.gate1 = SimpleGate()
        gated_channels = depthwise_channels // 2
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(gated_channels, gated_channels, 1),
        )
        self.conv2 = nn.Conv2d(gated_channels, channels, 1)
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))

        self.norm2 = LayerNorm2d(channels)
        self.conv3 = nn.Conv2d(channels, ffn_channels, 1)
        self.gate2 = SimpleGate()
        self.conv4 = nn.Conv2d(ffn_channels // 2, channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.depthwise(self.conv1(self.norm1(x)))
        y = self.gate1(y)
        y = y * self.sca(y)
        x = x + self.conv2(y) * self.beta
        y = self.gate2(self.conv3(self.norm2(x)))
        return x + self.conv4(y) * self.gamma


class NAFStage(nn.Sequential):
    def __init__(self, channels: int, blocks: int):
        super().__init__(*[NAFBlock(channels) for _ in range(blocks)])


class NAFNetSharedEncoder(nn.Module):
    """Four-scale NAFNet encoder used as the formal V1 backbone."""

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 32,
        encoder_blocks: tuple[int, int, int, int] = (2, 2, 4, 6),
    ):
        super().__init__()
        c = base_channels
        self.intro = nn.Conv2d(in_channels, c, 3, padding=1)
        self.stage1 = NAFStage(c, encoder_blocks[0])
        self.down1 = nn.Conv2d(c, c * 2, 2, stride=2)
        self.stage2 = NAFStage(c * 2, encoder_blocks[1])
        self.down2 = nn.Conv2d(c * 2, c * 4, 2, stride=2)
        self.stage3 = NAFStage(c * 4, encoder_blocks[2])
        self.down3 = nn.Conv2d(c * 4, c * 8, 2, stride=2)
        self.stage4 = NAFStage(c * 8, encoder_blocks[3])
        self.out_channels = [c, c * 2, c * 4, c * 8]

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        f1 = self.stage1(self.intro(x))
        f2 = self.stage2(self.down1(f1))
        f3 = self.stage3(self.down2(f2))
        f4 = self.stage4(self.down3(f3))
        return [f1, f2, f3, f4]


class NAFUpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, blocks: int = 2):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, out_channels, 1)
        self.skip_proj = nn.Conv2d(skip_channels, out_channels, 1)
        self.stage = NAFStage(out_channels, blocks)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.stage(self.proj(x) + self.skip_proj(skip))


class NAFNetRestorationDecoder(nn.Module):
    def __init__(
        self,
        encoder_channels: list[int],
        out_channels: int = 3,
        decoder_blocks: tuple[int, int, int] = (2, 2, 2),
    ):
        super().__init__()
        c1, c2, c3, c4 = encoder_channels
        self.up3 = NAFUpBlock(c4, c3, c3, decoder_blocks[0])
        self.up2 = NAFUpBlock(c3, c2, c2, decoder_blocks[1])
        self.up1 = NAFUpBlock(c2, c1, c1, decoder_blocks[2])
        self.refine = NAFStage(c1, 2)
        self.out = nn.Conv2d(c1, out_channels, 3, padding=1)

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        f1, f2, f3, f4 = features
        x = self.up3(f4, f3)
        x = self.up2(x, f2)
        x = self.up1(x, f1)
        return self.out(self.refine(x))
