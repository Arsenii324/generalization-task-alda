#!/usr/bin/env bash
# Full experiment matrix — 2 algos × 2 tasks × 3 seeds = 12 training runs,
# each followed by a post-hoc eval.py sweep over all distraction settings.
#
# Pipeline per run:
#   1. Train 500k steps (eval color-hard every 10k during training → wandb)
#   2. eval.py on final checkpoint: color easy/medium/hard + camera + background → wandb
#
# ─── Setup ────────────────────────────────────────────────────────────────────
# MUJOCO_GL=egl  (headless Linux GPU, fastest)
# DAVIS_PATH     path to extracted DAVIS 2017 dataset (video dirs directly inside)
#                Leave empty to skip background distraction in post-hoc eval.
#
# ─── Usage ────────────────────────────────────────────────────────────────────
# Sequential (one GPU):
#   MUJOCO_GL=egl bash scripts/run_cloud.sh
#
# With DAVIS background distraction in post-hoc eval:
#   MUJOCO_GL=egl DAVIS_PATH=~/datasets/DAVIS bash scripts/run_cloud.sh
#
# Parallel (all runs in background, use tmux or nohup):
#   MUJOCO_GL=egl bash scripts/run_cloud.sh --parallel
#
# Download DAVIS first (2.3 GB):
#   bash scripts/download_davis.sh ~/datasets/DAVIS
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

PARALLEL=false
for arg in "$@"; do [[ "$arg" == "--parallel" ]] && PARALLEL=true; done

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p logs

# ── Config ────────────────────────────────────────────────────────────────────
PROJECT="alda-sac-pixel"
TOTAL=500000
EVAL_FREQ=10000       # color-hard eval during training, every N steps
N_EVAL_TRAIN=10       # episodes per during-training eval
N_EVAL_POSTHOC=10     # episodes per setting in post-hoc eval
BUFFER_SIZE=100000    # reduce to 50000/20000 if RAM tight (~6 GB / ~2.4 GB)

TASKS=("cartpole-balance" "walker-walk")
SEEDS=(1 2 3)
ALGOS=("sac_ae" "alda")

# DAVIS path for background distraction in post-hoc eval (empty = skip background)
DAVIS_PATH="${DAVIS_PATH:-}"
# ──────────────────────────────────────────────────────────────────────────────

TRAIN_SHARED="--total-timesteps $TOTAL
  --eval-frequency $EVAL_FREQ
  --n-eval-episodes $N_EVAL_TRAIN
  --buffer-size $BUFFER_SIZE
  --track --wandb-project $PROJECT"

pids=()

# Find the most recent checkpoint for a given run_name prefix
latest_ckpt() {
  local prefix="$1"   # e.g. "sac_pixel__walker-walk__seed1"
  ls checkpoints/${prefix}__*/step_*.pt 2>/dev/null \
    | sort -t_ -k2 -n | tail -1
}

run_one() {
  local algo="$1" task="$2" seed="$3"
  local logfile="logs/${algo}__${task}__seed${seed}.log"

  echo "[start] $algo  task=$task  seed=$seed  log=$logfile"

  # ── Step 1: Train ──────────────────────────────────────────────────────────
  if [[ "$algo" == "sac_ae" ]]; then
    MUJOCO_GL=${MUJOCO_GL:-egl} uv run python train_sac_pixel.py \
      $TRAIN_SHARED --task "$task" --seed "$seed" \
      >> "$logfile" 2>&1
    CKPT_PREFIX="sac_pixel__${task}__seed${seed}"
  else
    MUJOCO_GL=${MUJOCO_GL:-egl} uv run python train_alda.py \
      $TRAIN_SHARED --task "$task" --seed "$seed" \
      >> "$logfile" 2>&1
    CKPT_PREFIX="alda__${task}__seed${seed}"
  fi

  # ── Step 2: Post-hoc full eval sweep ──────────────────────────────────────
  CKPT=$(latest_ckpt "$CKPT_PREFIX")
  if [[ -z "$CKPT" ]]; then
    echo "[warn] no checkpoint found for $CKPT_PREFIX — skipping eval"
    return
  fi
  echo "[eval] $algo $task seed=$seed  checkpoint=$CKPT"

  EVAL_ARGS="--checkpoint $CKPT
    --n-eval-episodes $N_EVAL_POSTHOC
    --distractions color camera
    --difficulties easy medium hard
    --track --wandb-project $PROJECT"

  if [[ -n "$DAVIS_PATH" ]]; then
    EVAL_ARGS="$EVAL_ARGS --distractions color camera background
               --background-dataset-path $DAVIS_PATH"
  fi

  MUJOCO_GL=${MUJOCO_GL:-egl} uv run python eval.py $EVAL_ARGS >> "$logfile" 2>&1
  echo "[done] $algo $task seed=$seed"
}

echo "============================================================"
echo " ALDA Experiment Matrix"
echo " algos : ${ALGOS[*]}"
echo " tasks : ${TASKS[*]}"
echo " seeds : ${SEEDS[*]}"
echo " steps : $TOTAL   eval_freq: $EVAL_FREQ"
echo " buffer: $BUFFER_SIZE slots"
echo " DAVIS : ${DAVIS_PATH:-not set, background distraction skipped}"
echo " parallel: $PARALLEL"
echo "============================================================"
echo ""

for TASK in "${TASKS[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    for ALGO in "${ALGOS[@]}"; do
      if $PARALLEL; then
        run_one "$ALGO" "$TASK" "$SEED" &
        pids+=($!)
      else
        run_one "$ALGO" "$TASK" "$SEED"
      fi
    done
  done
done

if $PARALLEL && [[ ${#pids[@]} -gt 0 ]]; then
  echo "Waiting for ${#pids[@]} runs (logs in ./logs/)..."
  fail=0
  for pid in "${pids[@]}"; do
    wait "$pid" || { echo "  PID $pid failed"; fail=1; }
  done
  [[ $fail -eq 0 ]] && echo "All runs completed." || echo "Some runs failed — check ./logs/"
fi

echo ""
echo "Results: https://wandb.ai/<entity>/$PROJECT"
