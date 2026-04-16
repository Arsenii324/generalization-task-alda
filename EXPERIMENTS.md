# Experimental Setup

**Goal:** Reproduce the key ALDA result — zero-shot OOD generalization without data augmentation —
and compare against the SAC+AE baseline using identical training conditions.

---

## 1. Algorithms

| Tag | Description | Status |
|---|---|---|
| `sac_ae` | SAC + CNN encoder + reconstruction loss (Yarats et al.) | Done (`train_sac_pixel.py`) |
| `alda` | SAC + QLAE (per-dim codebooks) + Softmax associative memory | To implement |

Only these two are in scope. Additional baselines (DARLA, RePo, SVEA) require separate repos
and are not included in this reproduction.

---

## 2. Tasks

All tasks are pixel-based DeepMind Control Suite via `dm_control/shimmy`.

| `--task` flag | Domain | n_z_paper | State dim | Difficulty |
|---|---|---|---|---|
| `cartpole-balance` | Cartpole | 12 | 5 | Easy — use as smoke-test / sanity check |
| `walker-walk` | Walker | 12 | 24 | Main benchmark in paper |
| `finger-spin` | Finger | 12 | 12 | Also in paper |
| `ball_in_cup-catch` | Ball-in-cup | 12 | 8 | Also in paper |

> `n_z_paper = 12` is used for ALDA across all tasks (from paper Appendix A).  
> SAC+AE uses `latent_dim = 50` (standard from Yarats et al.; paper doesn't change this).

---

## 3. Hyperparameters

### 3.1 Shared (both algorithms)

| Parameter | Value | Source |
|---|---|---|
| `total_timesteps` | 500,000 | Paper |
| `frame_stack` k | 3 | Paper / DMC standard |
| `image_size` | 84 × 84 | Paper / DMC standard |
| `batch_size` | 256 | Paper |
| `learning_starts` | 1,000 | Paper |
| `gamma` γ | 0.99 | Paper |
| `tau` τ (Polyak) | 0.005 | Paper |
| `policy_frequency` | 2 | Paper |
| `target_frequency` | 1 | Paper |
| `autotune_alpha` | True | — |
| `policy_lr` | 3e-4 | — |
| `q_lr` | 1e-3 | — |
| Seeds | 1, 2, 3 | 3 seeds per condition |

### 3.2 SAC+AE specific

| Parameter | Value | Source |
|---|---|---|
| `latent_dim` | 50 | Yarats et al. |
| `recon_lr` | 1e-3 | Yarats et al. |
| `buffer_size` | 100,000 | Memory budget (100k × 2 × 9 × 84 × 84 ≈ 12 GB) |

### 3.3 ALDA specific

| Parameter | Value | Source |
|---|---|---|
| `n_z` (codebook dims) | 12 | Paper Appendix A |
| `codebook_size` K | 10 per dim | Paper Appendix A |
| `beta` β (Hopfield temperature) | 10.0 | Paper Appendix A.4 |
| `lambda_enc` λ_θ | 0.1 | Paper (weight decay on encoder) |
| `lambda_dec` λ_φ | 0.1 | Paper (weight decay on decoder) |
| `assoc_loss_weight` | 1.0 | Paper Eq. 7 |
| `recon_loss_weight` | 1.0 | Paper Eq. 7 |
| `buffer_size` | 100,000 | Same as SAC+AE |
| Encoder input | (B, 3, 84, 84) per frame | Each frame encoded separately |
| Policy input dim | k × n_z = 36 | 3 frames × 12 latents, concatenated |

**ALDA encoder note:** Unlike SAC+AE which encodes the full (9, 84, 84) stacked tensor at once,
ALDA encodes each of the k=3 frames as (3, 84, 84) → 12-dim latent, then concatenates to get
a 36-dim policy input. This matches the paper's temporal stacking description.

---

## 4. OOD Evaluation Protocol

Evaluation is run **during training** (every `eval_frequency` steps) and **post-hoc** via `eval.py`.

### 4.1 Settings tested

| Setting | Distraction | Difficulty | Extra data? | Priority |
|---|---|---|---|---|
| `clean` | none | — | No | Always run |
| `color-easy` | color | easy | No | High |
| `color-medium` | color | medium | No | High |
| `color-hard` | color | hard | No | High (main result) |
| `camera-easy` | camera | easy | No | Medium |
| `camera-medium` | camera | medium | No | Medium |
| `camera-hard` | camera | hard | No | Medium |
| `background-easy` | background | easy | DAVIS 2017 (~6 GB) | Low (skip if no dataset) |
| `background-hard` | background | hard | DAVIS 2017 (~6 GB) | Low |

Run at minimum: clean + color (easy/medium/hard). Camera adds coverage without extra data.

### 4.2 During-training evaluation (in `train_sac_pixel.py` / `train_alda.py`)

- Frequency: every 10,000 steps
- Episodes: 5 (fast, to not slow training)
- Setting: **color-hard only** (the hardest OOD, most signal)
- Logged to wandb: `eval/ood_color_hard_return`

### 4.3 Post-hoc evaluation (via `eval.py`)

- After training completes (or on best checkpoint), run full sweep
- Episodes per setting: 10
- Covers all enabled settings in Section 4.1

```bash
MUJOCO_GL=glfw uv run python eval.py \
  --checkpoint checkpoints/<run>/step_0500000.pt \
  --n-eval-episodes 10 \
  --distractions color camera \
  --difficulties easy medium hard \
  --track
```

---

## 5. Metrics

### 5.1 Primary metrics (reported per task, per algorithm)

| Metric | Formula | Interpretation |
|---|---|---|
| **Train return** | Mean episodic return on clean DMC | Measures task mastery |
| **Clean eval return** | Mean return on clean DMC at eval time (deterministic) | Same domain; policy quality |
| **OOD return** | Mean return on DistractingCS setting | Generalization |
| **Generalization gap** | (Clean return − OOD return) / Clean return × 100% | % drop when going OOD |
| **OOD std** | Std of episodic returns over 10 episodes | Stability under distraction |

All returns averaged over last 10 eval points (steps 410k–500k) for final table.

### 5.2 Training diagnostics (logged every 1,000 steps)

| Metric | Key | Alarm if... |
|---|---|---|
| Q-function loss | `losses/qf_loss` | Diverges (>1e4) or NaN |
| Reconstruction loss | `losses/recon_loss` | Doesn't decrease below 0.01 by step 50k |
| Actor loss | `losses/actor_loss` | NaN |
| Alpha (entropy temp) | `losses/alpha` | Collapses to 0 or explodes >10 |
| Steps-per-second | `charts/SPS` | < 5 on MPS (GPU may be unused) |
| **ALDA only:** Association loss | `losses/assoc_loss` | Doesn't decrease; may indicate codebook collapse |
| **ALDA only:** Codebook usage | `losses/codebook_usage` | < 50% means codebook collapse |

### 5.3 Final result table format

```
Task             | Algo    | Clean        | Color-Easy   | Color-Hard   | Gap (hard)
cartpole-balance | SAC+AE  | 850 ± 30     | 750 ± 50     | 600 ± 80     | 29%
cartpole-balance | ALDA    | 820 ± 40     | 800 ± 30     | 780 ± 40     |  5%
walker-walk      | SAC+AE  | 700 ± 60     | ...          | ...          | ...
walker-walk      | ALDA    | 680 ± 50     | ...          | ...          | ...
```
(Expected values are illustrative; fill in from actual runs.)

---

## 6. Wandb Organization

**Project:** `alda-sac-pixel` (same for both algorithms — enables direct comparison)

**Run naming convention:**
```
{algo}__{task}__seed{seed}__{unix_timestamp}
e.g.
  sac_ae__walker-walk__seed1__1776243915
  alda__walker-walk__seed1__1776259342
```

**Tags to set on each run:**
```python
wandb.init(
    tags=[algo, task, f"seed{seed}"],
    group=f"{algo}__{task}",   # groups seeds together in wandb UI
    ...
)
```

**Key wandb views to create:**
1. Group by `algo` + `task`, plot `eval/ood_color_hard_return` vs step (mean ± std across seeds)
2. Final bar chart: `eval/ood_color_hard_return` at step 500k, grouped by algo per task
3. Training curves: `charts/train_return` per task (check both algos learn the task)

---

## 7. Run Matrix

Full run matrix (2 algos × 4 tasks × 3 seeds = **24 runs**):

| Priority | Algo | Task | Seeds | Justification |
|---|---|---|---|---|
| 1 | `sac_ae` | `cartpole-balance` | 1,2,3 | Sanity check, fast |
| 2 | `sac_ae` | `walker-walk` | 1,2,3 | Main paper task |
| 3 | `alda` | `cartpole-balance` | 1,2,3 | Validate ALDA works |
| 4 | `alda` | `walker-walk` | 1,2,3 | Main ALDA result |
| 5 | `sac_ae` | `finger-spin` | 1,2,3 | Breadth |
| 6 | `alda` | `finger-spin` | 1,2,3 | Breadth |
| 7 | `sac_ae` | `ball_in_cup-catch` | 1,2,3 | Breadth |
| 8 | `alda` | `ball_in_cup-catch` | 1,2,3 | Breadth |

**Minimum viable result** (for a credible report): Priority 1–4, i.e., 12 runs on 2 tasks.

---

## 8. Launch Commands

### SAC+AE (cloud, full run)
```bash
MUJOCO_GL=glfw uv run python train_sac_pixel.py \
  --task walker-walk --seed 1 --total-timesteps 500000 \
  --eval-frequency 10000 --n-eval-episodes 5 \
  --track --wandb-project alda-sac-pixel
```

### ALDA (cloud, full run) — once `train_alda.py` is written
```bash
MUJOCO_GL=glfw uv run python train_alda.py \
  --task walker-walk --seed 1 --total-timesteps 500000 \
  --n-z 12 --codebook-size 10 --beta 10.0 \
  --lambda-enc 0.1 --lambda-dec 0.1 \
  --eval-frequency 10000 --n-eval-episodes 5 \
  --track --wandb-project alda-sac-pixel
```

### Local smoke test (≤2k steps, always use these first)
```bash
MUJOCO_GL=glfw uv run python train_sac_pixel.py \
  --task cartpole-balance --seed 1 --total-timesteps 2000 \
  --learning-starts 500 --eval-frequency 1000 --n-eval-episodes 2 --track

MUJOCO_GL=glfw uv run python train_alda.py \
  --task cartpole-balance --seed 1 --total-timesteps 2000 \
  --learning-starts 500 --eval-frequency 1000 --n-eval-episodes 2 --track
```

---

## 9. Sanity Checks

Run these before launching full experiments:

| Check | How | Pass condition |
|---|---|---|
| DMC env returns correct obs shape | `python -c "from envs import make_dmc; e=make_dmc('walker-walk'); o,_=e.reset(); print(o.shape)"` | `(9, 84, 84)` |
| DistractingCS color loads | `python -c "from envs import make_dcs; e=make_dcs('walker_walk', 'color', 'hard'); o,_=e.reset(); print(o.shape)"` | `(9, 84, 84)` |
| Encoder forward pass | `python -c "from alda.encoder import PixelEncoder; import torch; e=PixelEncoder(); z=e(torch.randn(2,9,84,84)); print(z.shape)"` | `torch.Size([2, 50])` |
| QLAE quantization (to add) | Unit test: codebook usage > 50% after 1k random inputs | Passes |
| Train 2k steps, no NaN | Smoke test script above | Runs to completion, losses printed |
| Checkpoint loads in eval.py | `eval.py --checkpoint <ckpt> --n-eval-episodes 2` | Prints result table |

---

## 10. Known Limitations and Risk Mitigations

| Risk | Mitigation |
|---|---|
| Codebook collapse in QLAE | Monitor `codebook_usage` metric; use commitment weight 0.25 if collapse observed |
| QLAE latent dim 12 too small for complex tasks | Try 20 if `walker-walk` fails to learn; paper says 12 suffices |
| DistractingCS `camera` env crashes | Catch exception in `eval.py`; log NaN for that setting |
| Background distraction needs DAVIS dataset | Skip by default; run separately if dataset is available |
| MPS slower than CUDA for conv | Use cloud GPU for 500k runs; MPS only for smoke tests |
| SAC+AE and ALDA not directly comparable if different latent dims | Keep `latent_dim=12` for SAC+AE too in ablation table; main comparison uses respective best settings |
| Cartpole trivially solvable (too easy) | Expect generalization gap to be small; use `walker-walk` as primary metric |
