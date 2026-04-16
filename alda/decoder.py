"""CNN decoder mirroring PixelEncoder."""

import torch
import torch.nn as nn


class PixelDecoder(nn.Module):
    """Transpose-conv decoder.

    Input:  (B, latent_dim)
    Output: (B, C*k, 84, 84) float32

    Spatial progression (mirrors encoder):
        35 → 37 → 39 → 41 → 84  (output_padding=1 on last layer)
    """

    CONV_FLAT = 32 * 35 * 35  # 39 200

    def __init__(self, out_channels: int = 9, latent_dim: int = 50):
        super().__init__()
        self.fc = nn.Linear(latent_dim, self.CONV_FLAT)
        self.deconvs = nn.Sequential(
            nn.ConvTranspose2d(32, 32, kernel_size=3, stride=1), nn.ReLU(),
            nn.ConvTranspose2d(32, 32, kernel_size=3, stride=1), nn.ReLU(),
            nn.ConvTranspose2d(32, 32, kernel_size=3, stride=1), nn.ReLU(),
            # output_padding=1 recovers the exact 84 px lost to floor in encoder stride-2 conv
            nn.ConvTranspose2d(32, out_channels, kernel_size=3, stride=2, output_padding=1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.fc(z)).view(z.size(0), 32, 35, 35)
        return self.deconvs(h)
