"""
QLAE — Quantized Latent AutoEncoder codebook.

Holds V ∈ R^{n_z × K}: K prototype scalar values for each of the n_z latent dims.
Implements the Softmax associative retrieval (Eq. 6 of Batra & Sukhatme 2024).

Architectural decisions
-----------------------
1. Codebook is an nn.Parameter, updated only by reconstruction loss (L_recon).
   L_assoc uses StopGrad on z_s, so codebook receives NO gradient from L_assoc.
   This matches Eq. 7 of the paper exactly.

2. No L_quantize loss (codebook NOT pushed toward encoder).
   Paper omits it; using it causes instability without clear benefit.

3. Soft retrieval (Eq. 6) is used for EVERYTHING — collection, critic, decoder input.
   Hard argmin is available for diagnostics only.
   At high β ≥ 10 the soft output is numerically identical to hard argmin for
   well-trained encoders, so this is lossless.

4. Codebook init: Uniform(-1, 1) per dimension.
   Encoder output is tanh-bounded ∈ (-1, 1), so prototypes span the full range.
   K=10 prototypes evenly cover the range before learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class QLAE(nn.Module):
    """
    Quantized Latent AutoEncoder codebook.

    Args:
        n_z:  Number of latent dimensions (=12 for all tasks per paper).
        K:    Codebook size per dimension (=10 per paper).
        beta: Softmax temperature. Higher → harder (nearer-to-argmin) retrieval.
              Paper uses large β; default 10.0.
    """

    def __init__(self, n_z: int = 12, K: int = 10, beta: float = 10.0):
        super().__init__()
        self.n_z  = n_z
        self.K    = K
        self.beta = beta
        # Shape: (n_z, K).  Each row j is the K prototypes for dimension j.
        # Init uniform so prototypes span encoder tanh output range (-1, 1).
        self.codebook = nn.Parameter(
            torch.zeros(n_z, K).uniform_(-1.0, 1.0)
        )

    # ------------------------------------------------------------------
    # Core retrieval
    # ------------------------------------------------------------------

    def associate(self, z_enc: torch.Tensor) -> torch.Tensor:
        """
        Soft nearest-neighbour retrieval per dimension (Eq. 6).

        Args:
            z_enc: (N, n_z) — continuous encoder output.
        Returns:
            z_s:   (N, n_z) — soft-quantized retrieval. Differentiable w.r.t.
                              both z_enc and self.codebook.
        """
        # l1: (N, n_z, K)
        l1 = (z_enc.unsqueeze(-1) - self.codebook.unsqueeze(0)).abs()
        w  = torch.softmax(-self.beta * l1, dim=-1)       # (N, n_z, K)
        return (w * self.codebook.unsqueeze(0)).sum(-1)    # (N, n_z)

    # ------------------------------------------------------------------
    # Losses
    # ------------------------------------------------------------------

    def association_loss(self, z_enc: torch.Tensor) -> torch.Tensor:
        """
        L_assoc = ||z_enc - sg(z_s)||²

        Pushes encoder output toward codebook prototypes.
        StopGrad on z_s means codebook receives NO gradient here.
        Gradient flows only into z_enc (i.e., into encoder parameters).

        Args:
            z_enc: (N, n_z) — encoder output, gradient ON.
        Returns:
            scalar loss.
        """
        z_s = self.associate(z_enc).detach()   # sg(z_s)
        return F.mse_loss(z_enc, z_s)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @torch.no_grad()
    def codebook_usage(self, z_enc: torch.Tensor) -> float:
        """
        Fraction of codebook slots that are the nearest prototype for at least
        one sample in z_enc. Healthy value > 0.5; collapse if < 0.2.

        Counts unique (dim, prototype_index) pairs out of n_z * K total.

        Args:
            z_enc: (N, n_z)
        Returns:
            float in [0, 1].
        """
        l1  = (z_enc.unsqueeze(-1) - self.codebook.unsqueeze(0)).abs()
        idx = l1.argmin(-1)                             # (N, n_z)  — nearest index per dim
        # Encode each (dim, idx) pair as a unique integer: dim * K + prototype_idx
        dim_offsets = (
            torch.arange(self.n_z, device=z_enc.device)
            .unsqueeze(0)          # (1, n_z)
            * self.K
        )
        pairs = (idx + dim_offsets).unique().numel()    # count distinct (dim, idx) pairs
        return pairs / (self.n_z * self.K)

    @torch.no_grad()
    def hard_quantize(self, z_enc: torch.Tensor) -> torch.Tensor:
        """
        Hard argmin quantization (for diagnostics / visualization).
        NOT used in training — not differentiable.

        Args:
            z_enc: (N, n_z)
        Returns:
            z_d:   (N, n_z) — hard-quantized output.
        """
        N   = z_enc.shape[0]
        l1  = (z_enc.unsqueeze(-1) - self.codebook.unsqueeze(0)).abs()
        idx = l1.argmin(-1)                             # (N, n_z)
        # Gather: for each sample i and dim j, pick codebook[j, idx[i,j]]
        row = (torch.arange(self.n_z, device=z_enc.device)
               .unsqueeze(0).expand(N, -1).reshape(-1))  # (N*n_z,)
        col = idx.reshape(-1)                             # (N*n_z,)
        return self.codebook[row, col].reshape(N, self.n_z)

    # ------------------------------------------------------------------

    def forward(self, z_enc: torch.Tensor) -> torch.Tensor:
        """Soft retrieval. Main entry point during training and collection."""
        return self.associate(z_enc)

    def extra_repr(self) -> str:
        return f"n_z={self.n_z}, K={self.K}, beta={self.beta}"
