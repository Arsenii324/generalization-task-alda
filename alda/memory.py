"""
Associative memory interface for eval-time OOD remapping.

In ALDA, the QLAE codebook IS the associative memory — no separate module needed.
This file provides a clean remap() function used by eval.py and evaluate_ood().

At test time:
    1. OOD frame is encoded by f_θ → z_enc (may be OOD in some dimensions)
    2. remap(z_enc, qlae) soft-maps each dimension to the nearest codebook prototype
    3. z_remapped is passed to the policy instead of z_enc
    4. Because dimensions are independent, only OOD dims are corrected;
       task-relevant dims that are already in-distribution map to themselves.
"""

import torch
from alda.qlae import QLAE


@torch.no_grad()
def remap(z_enc: torch.Tensor, qlae: QLAE) -> torch.Tensor:
    """
    OOD correction via QLAE associative retrieval.

    Args:
        z_enc: (N, n_z) — raw encoder output on (possibly OOD) frames.
        qlae:  trained QLAE module with learned codebook.
    Returns:
        z_s:   (N, n_z) — in-distribution approximation.
    """
    return qlae.associate(z_enc)
