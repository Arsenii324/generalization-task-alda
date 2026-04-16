#!/usr/bin/env bash
# Full end-to-end validation including background distraction.
#
# What is tested:
#   1. SAC+AE training (600 steps) — losses, color-hard OOD eval, checkpoint, wandb
#   2. ALDA training   (600 steps) — same + assoc_loss, cb_usage
#   3. eval.py on both checkpoints — color easy/hard + camera + background (fake DAVIS)
#
# Expected time: ~3-4 min on MPS, <1 min on a cloud GPU (EGL).
#
# Usage:
#   MUJOCO_GL=glfw bash scripts/test_quick.sh
#   MUJOCO_GL=glfw bash scripts/test_quick.sh --no-track
set -euo pipefail

TRACK="--track"
for arg in "$@"; do [[ "$arg" == "--no-track" ]] && TRACK=""; done

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

FAKE_DAVIS="/tmp/fake_davis_test"
WANDB_PROJECT="alda-sac-pixel"

TRAIN_ARGS="--task cartpole-balance --seed 42
  --total-timesteps 600
  --learning-starts 200
  --buffer-size 2000
  --batch-size 64
  --eval-frequency 500
  --n-eval-episodes 2
  $TRACK
  --wandb-project $WANDB_PROJECT"

echo "============================================================"
echo " End-to-end test: training + OOD eval incl. background"
echo "============================================================"

# ── Step 0: Fake DAVIS dataset ────────────────────────────────────────────────
echo ""
echo "--- Creating fake DAVIS dataset (8 videos × 30 frames) ---"
uv run python scripts/make_fake_davis.py --out-dir "$FAKE_DAVIS" --n-videos 60 --n-frames 30

# ── Step 1: Train SAC+AE ──────────────────────────────────────────────────────
echo ""
echo "--- Training SAC+AE (600 steps, 1 OOD eval) ---"
MUJOCO_GL=${MUJOCO_GL:-glfw} uv run python train_sac_pixel.py $TRAIN_ARGS

# ── Step 2: Train ALDA ────────────────────────────────────────────────────────
echo ""
echo "--- Training ALDA (600 steps, 1 OOD eval) ---"
MUJOCO_GL=${MUJOCO_GL:-glfw} uv run python train_alda.py $TRAIN_ARGS

# ── Step 3: Post-hoc eval on both checkpoints ────────────────────────────────
echo ""
echo "--- Running post-hoc eval.py (color + camera + background) ---"

find checkpoints -name "step_*.pt" -newer "$FAKE_DAVIS" | sort | while read CKPT; do
  echo ""
  echo "  Evaluating: $CKPT"
  MUJOCO_GL=${MUJOCO_GL:-glfw} uv run python eval.py \
    --checkpoint "$CKPT" \
    --n-eval-episodes 2 \
    --distractions color camera background \
    --difficulties easy hard \
    --background-dataset-path "$FAKE_DAVIS" \
    $TRACK \
    --wandb-project "$WANDB_PROJECT"
done

echo ""
echo "============================================================"
echo " Test passed. All pipeline stages completed:"
echo "   SAC+AE training + color-hard eval during training ✓"
echo "   ALDA training   + color-hard eval during training ✓"
echo "   eval.py: color + camera + background (fake DAVIS)  ✓"
echo " Check wandb: $WANDB_PROJECT"
echo "============================================================"
