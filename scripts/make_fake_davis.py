"""
Create a minimal synthetic DAVIS-format dataset for local testing.

Generates 8 video directories (enough for 'medium' difficulty, which needs 8;
'easy' needs 4, 'hard' needs all 60 — skip hard locally).
Each video has 30 frames of solid random colors as JPEG images.

The distracting_control background module reads from:
  <dataset_path>/<video_name>/<frame>.jpg
where video names come from DAVIS17_TRAINING_VIDEOS[:8] for 'train' split.

Usage:
    uv run python scripts/make_fake_davis.py [--out-dir /tmp/fake_davis]
"""

import argparse
import os
from pathlib import Path

import numpy as np
from PIL import Image

from distracting_control.background import DAVIS17_TRAINING_VIDEOS


def make_fake_davis(out_dir: Path, n_videos: int = 8, n_frames: int = 30,
                    width: int = 480, height: int = 270) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Use the first n_videos from DAVIS training list — same names the module will look for
    videos = DAVIS17_TRAINING_VIDEOS[:n_videos]
    print(f"Creating {n_videos} fake videos × {n_frames} frames at {width}×{height}px")
    print(f"  → {out_dir}/")

    rng = np.random.default_rng(42)
    for video in videos:
        vdir = out_dir / video
        vdir.mkdir(exist_ok=True)
        # Each video gets a distinct solid hue with slight per-frame variation
        base_color = rng.integers(0, 256, size=3, dtype=np.uint8)
        for i in range(n_frames):
            noise   = rng.integers(-15, 15, size=3).clip(-base_color, 255 - base_color)
            color   = (base_color + noise).astype(np.uint8)
            frame   = np.full((height, width, 3), color, dtype=np.uint8)
            img     = Image.fromarray(frame)
            img.save(vdir / f"{i:05d}.jpg", quality=85)
        print(f"  {video}/  ({n_frames} frames)")

    print(f"\nDone. Use as: --background-dataset-path {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="/tmp/fake_davis")
    parser.add_argument("--n-videos", type=int, default=60,
                        help="60 = all training videos (required for hard difficulty); 8 for medium; 4 for easy")
    parser.add_argument("--n-frames", type=int, default=30)
    args = parser.parse_args()
    make_fake_davis(Path(args.out_dir), args.n_videos, args.n_frames)
