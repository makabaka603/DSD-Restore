import torch
from torch import nn

from .backbone import NAFNetRestorationDecoder, NAFNetSharedEncoder


class NAFNetBaseline(nn.Module):
    """Controlled NAFNet-style baseline built from the repository backbone."""

    supports_dsd_diagnostics = False

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_channels: int = 32,
        **_: object,
    ):
        super().__init__()
        self.encoder = NAFNetSharedEncoder(
            in_channels=in_channels,
            base_channels=base_channels,
        )
        self.decoder = NAFNetRestorationDecoder(
            self.encoder.out_channels,
            out_channels=out_channels,
        )

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.encoder(image)
        residual = self.decoder(features)
        restored = image + torch.tanh(residual)
        return {"restored": restored}
