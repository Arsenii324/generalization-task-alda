#!/usr/bin/env bash
# One-time setup for Kaggle GPU notebooks.
# Run this in a notebook shell cell before anything else.
#
# Assumes:
#   - GPU accelerator is ON (Settings → Accelerator → GPU T4 x2 or P100)
#   - Internet is ON (Settings → Internet → On)
#   - WANDB_API_KEY is added as a Kaggle Secret (see below)
#
# After this script completes, run the experiments with:
#   MUJOCO_GL=egl bash scripts/test_quick.sh   # ~30 sec on T4
#   MUJOCO_GL=egl bash scripts/run_cloud.sh    # full matrix
set -euo pipefail

echo "=== 1. System packages for MuJoCo EGL headless rendering ==="
apt-get install -y --quiet \
  libgl1-mesa-dev \
  libglew-dev \
  libosmesa6-dev \
  libglfw3 \
  patchelf

echo "=== 2. Install uv ==="
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
echo "uv $(uv --version)"

echo "=== 3. Install Python 3.12 and sync dependencies ==="
# uv will download Python 3.12 automatically (Kaggle base is 3.10, which is below our >=3.11)
uv python install 3.12
uv sync
echo "Dependencies installed."

echo ""
echo "=== Done. Next steps ==="
echo "  1. Set WANDB key (see instructions below)"
echo "  2. MUJOCO_GL=egl bash scripts/test_quick.sh"
echo "  3. MUJOCO_GL=egl bash scripts/run_cloud.sh"
