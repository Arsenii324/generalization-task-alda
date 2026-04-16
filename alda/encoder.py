"""CNN encoder for 84×84 pixel observations."""

import torch
import torch.nn as nn


class PixelEncoder(nn.Module):
    """4-layer CNN encoder.

    Input:  (B, C*k, 84, 84) float32, normalised to [-0.5, 0.5]
    Output: (B, latent_dim) float32

    Conv spatial progression (84×84 input, no padding):
        Conv(s=2): 84 → 41
        Conv(s=1): 41 → 39 → 37 → 35
        Flatten:   32 × 35 × 35 = 39 200
    """

    CONV_FLAT = 32 * 35 * 35  # 39 200

    def __init__(self, in_channels: int = 9, latent_dim: int = 50):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2), nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1), nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1), nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1), nn.ReLU(),
        )
        self.fc = nn.Linear(self.CONV_FLAT, latent_dim)
        self.ln = nn.LayerNorm(latent_dim)

    def forward(self, obs: torch.Tensor, detach: bool = False) -> torch.Tensor:
        """
        Args:
            obs:    (B, C*k, 84, 84) float32
            detach: stop gradient at the output (used for actor updates)
        """
        h = self.convs(obs).view(obs.size(0), -1)
        h = torch.tanh(self.ln(self.fc(h)))
        return h.detach() if detach else h
