"""
DMC and Distracting Control Suite environment wrappers.

Produces pixel observations of shape (C*k, H, W) = (9, 84, 84) by default
(3 RGB channels × 3 stacked frames), normalized to float32 in [-0.5, 0.5].

Usage:
    from envs import make_dmc, make_dcs
    env = make_dmc("walker-walk")
    env = make_dcs("walker-walk", distraction="color", difficulty="hard")
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import shimmy  # noqa: F401 — registers dm_control envs with gymnasium


# ---------------------------------------------------------------------------
# Internal wrappers
# ---------------------------------------------------------------------------

class PixelObservationWrapper(gym.ObservationWrapper):
    """Replaces dict/proprioceptive obs with rendered RGB pixels."""

    def __init__(self, env: gym.Env, image_size: int = 84):
        super().__init__(env)
        self.image_size = image_size
        self.observation_space = spaces.Box(
            low=0, high=255,
            shape=(3, image_size, image_size),
            dtype=np.uint8,
        )

    def observation(self, obs):
        # dm_control envs support render() via shimmy
        img = self.env.render()  # (H, W, 3) uint8
        img = _resize(img, self.image_size)  # (84, 84, 3)
        return img.transpose(2, 0, 1)  # (3, 84, 84)


class FrameStackWrapper(gym.Wrapper):
    """Stacks k most recent frames into a (C*k, H, W) observation."""

    def __init__(self, env: gym.Env, k: int = 3):
        super().__init__(env)
        self.k = k
        self._frames: list[np.ndarray] = []
        c, h, w = env.observation_space.shape
        self.observation_space = spaces.Box(
            low=0, high=255,
            shape=(c * k, h, w),
            dtype=np.uint8,
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._frames = [obs] * self.k
        return self._stack(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._frames.pop(0)
        self._frames.append(obs)
        return self._stack(), reward, terminated, truncated, info

    def _stack(self) -> np.ndarray:
        return np.concatenate(self._frames, axis=0)  # (C*k, H, W)


class NormalizePixels(gym.ObservationWrapper):
    """Converts uint8 (C, H, W) to float32 in [-0.5, 0.5]."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        obs_space = env.observation_space
        self.observation_space = spaces.Box(
            low=-0.5, high=0.5,
            shape=obs_space.shape,
            dtype=np.float32,
        )

    def observation(self, obs: np.ndarray) -> np.ndarray:
        return obs.astype(np.float32) / 255.0 - 0.5


# ---------------------------------------------------------------------------
# Public factory functions
# ---------------------------------------------------------------------------

def make_dmc(
    task: str,
    seed: int = 0,
    image_size: int = 84,
    frame_stack: int = 3,
    normalize: bool = True,
) -> gym.Env:
    """
    Create a DMC environment with pixel observations.

    Args:
        task: dm_control task name, e.g. "walker-walk", "cartpole-balance",
              "ball_in_cup-catch", "finger-spin"
        seed: random seed
        image_size: square pixel size (default 84)
        frame_stack: number of frames to stack (default 3)
        normalize: if True, pixels are float32 in [-0.5, 0.5]; else uint8

    Returns:
        Wrapped gymnasium environment with obs shape (9, 84, 84).
    """
    env_id = f"dm_control/{task}-v0"
    env = gym.make(env_id, render_mode="rgb_array")
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = PixelObservationWrapper(env, image_size=image_size)
    env = FrameStackWrapper(env, k=frame_stack)
    if normalize:
        env = NormalizePixels(env)
    env.action_space.seed(seed)
    return env


def make_dcs(
    task: str,
    distraction: str = "color",
    difficulty: str = "hard",
    seed: int = 0,
    image_size: int = 84,
    frame_stack: int = 3,
    normalize: bool = True,
    background_dataset_path: str | None = None,
) -> gym.Env:
    """
    Create a Distracting Control Suite environment with pixel observations.

    Requires `distracting_control` to be installed:
        uv add distracting-control

    For background distraction, also requires the DAVIS 2017 dataset (~6GB).
    Set background_dataset_path to the path of the extracted dataset.

    Args:
        task: DMC task name with underscore, e.g. "walker_walk"
        distraction: one of "color", "camera", "background"
        difficulty: one of "easy", "medium", "hard"
        seed: random seed
        image_size: square pixel size
        frame_stack: number of frames to stack
        normalize: if True, pixels are float32 in [-0.5, 0.5]
        background_dataset_path: path to DAVIS 2017 dataset (background distraction only)
    """
    _patch_mujoco3_tex_rgb()  # no-op on mujoco 2.x
    from distracting_control import suite as dcs_suite

    domain, task_name = task.split("_", 1)

    raw_env = dcs_suite.load(
        domain_name=domain,
        task_name=task_name,
        difficulty=difficulty,
        distraction_types=[distraction],
        distraction_seed=seed,
        background_dataset_path=background_dataset_path,
        task_kwargs={"random": seed},
        from_pixels=False,   # we render manually for consistent image_size
        pixels_only=False,
    )

    env = _DCSWrapper(raw_env, image_size=image_size)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = FrameStackWrapper(env, k=frame_stack)
    if normalize:
        env = NormalizePixels(env)
    return env


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resize(img: np.ndarray, size: int) -> np.ndarray:
    """Resize HxWxC image to size×size using nearest-neighbour (no cv2 dep)."""
    from PIL import Image
    return np.array(Image.fromarray(img).resize((size, size), Image.BILINEAR))


class _DCSWrapper(gym.Env):
    """Minimal gymnasium wrapper around a raw dm_control environment."""

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, env, image_size: int = 84):
        self._env = env
        self._image_size = image_size
        action_spec = env.action_spec()
        self.action_space = spaces.Box(
            low=action_spec.minimum.astype(np.float32),
            high=action_spec.maximum.astype(np.float32),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=0, high=255,
            shape=(3, image_size, image_size),
            dtype=np.uint8,
        )
        self.render_mode = "rgb_array"

    def reset(self, *, seed=None, options=None):
        ts = self._env.reset()
        return self._render_obs(), {}

    def step(self, action):
        ts = self._env.step(action)
        obs = self._render_obs()
        reward = float(ts.reward) if ts.reward is not None else 0.0
        terminated = ts.last()
        truncated = False
        return obs, reward, terminated, truncated, {}

    def _render_obs(self) -> np.ndarray:
        img = self._env.physics.render(
            height=self._image_size, width=self._image_size, camera_id=0
        )
        return img.transpose(2, 0, 1)  # (3, H, W)

    def render(self):
        return self._env.physics.render(
            height=self._image_size, width=self._image_size, camera_id=0
        )

    def close(self):
        self._env.close()


# ---------------------------------------------------------------------------
# MuJoCo 3.x compatibility
# ---------------------------------------------------------------------------

_TEX_RGB_PATCHED = False

def _patch_mujoco3_tex_rgb() -> None:
    """
    Add tex_rgb as a property alias for tex_data on dm_control's MjModel.

    MuJoCo 2.x exposed texture pixel data as model.tex_rgb (uint8 RGB array).
    MuJoCo 3.x renamed this to model.tex_data with the same layout.
    distracting_control.background uses tex_rgb, so it breaks on MuJoCo 3.x.

    This patch is applied once at import time and is a no-op on MuJoCo 2.x.
    """
    global _TEX_RGB_PATCHED
    if _TEX_RGB_PATCHED:
        return

    try:
        from dm_control.mujoco.wrapper.core import MjModel
        if not hasattr(MjModel, "tex_rgb"):
            # tex_data has identical layout (flat uint8 array of all texture pixels)
            MjModel.tex_rgb = property(
                fget=lambda self: self.tex_data,
                fset=lambda self, v: setattr(self, "tex_data", v),
            )
        _TEX_RGB_PATCHED = True
    except Exception:
        pass  # silently skip if dm_control not available
