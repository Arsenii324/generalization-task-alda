"""Pixel replay buffer — stores uint8, returns float32."""

from typing import Tuple

import numpy as np
import torch


class PixelReplayBuffer:
    """Circular replay buffer for pixel observations.

    Observations are stored as uint8 [0, 255] and converted to
    float32 [-0.5, 0.5] on sample, saving ~4x RAM vs float32 storage.

    Memory footprint (obs + next_obs):
        capacity × 2 × C × H × W bytes
        e.g. 100k × 2 × 9 × 84 × 84 ≈ 12 GB
             50k  × 2 × 9 × 84 × 84 ≈  6 GB
             20k  × 2 × 9 × 84 × 84 ≈  2.4 GB
    """

    def __init__(
        self,
        obs_shape: tuple,   # (C*k, H, W), e.g. (9, 84, 84)
        act_dim: int,
        capacity: int,
        device: torch.device,
    ):
        self.capacity = capacity
        self.device = device
        self._ptr = 0
        self._size = 0

        self._obs      = np.empty((capacity, *obs_shape), dtype=np.uint8)
        self._next_obs = np.empty((capacity, *obs_shape), dtype=np.uint8)
        self._actions  = np.empty((capacity, act_dim),    dtype=np.float32)
        self._rewards  = np.empty((capacity,),            dtype=np.float32)
        self._dones    = np.empty((capacity,),            dtype=np.float32)

        mem_gb = self._obs.nbytes * 2 / 1024**3
        print(f"[PixelReplayBuffer] capacity={capacity:,}  obs={obs_shape}  "
              f"RAM≈{mem_gb:.1f} GB (obs+next_obs, uint8)")

    # ------------------------------------------------------------------

    def add(
        self,
        obs: np.ndarray,       # float32 [-0.5, 0.5], shape obs_shape
        next_obs: np.ndarray,
        action: np.ndarray,    # float32, shape (act_dim,)
        reward: float,
        done: float,
    ) -> None:
        self._obs[self._ptr]      = _f2u(obs)
        self._next_obs[self._ptr] = _f2u(next_obs)
        self._actions[self._ptr]  = action.astype(np.float32)
        self._rewards[self._ptr]  = reward
        self._dones[self._ptr]    = done
        self._ptr  = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        idx = np.random.randint(0, self._size, size=batch_size)
        return (
            _u2ft(self._obs[idx],      self.device),
            _u2ft(self._next_obs[idx], self.device),
            torch.as_tensor(self._actions[idx], device=self.device),
            torch.as_tensor(self._rewards[idx], device=self.device),
            torch.as_tensor(self._dones[idx],   device=self.device),
        )

    def __len__(self) -> int:
        return self._size


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _f2u(obs: np.ndarray) -> np.ndarray:
    """float32 [-0.5, 0.5] → uint8 [0, 255]"""
    return np.round((obs + 0.5) * 255).clip(0, 255).astype(np.uint8)


def _u2ft(obs: np.ndarray, device: torch.device) -> torch.Tensor:
    """uint8 [0, 255] → float32 tensor [-0.5, 0.5]"""
    return torch.as_tensor(obs, dtype=torch.float32, device=device) / 255.0 - 0.5
