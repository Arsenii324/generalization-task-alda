# ALDA Paper Notes

**Zero-Shot Generalization of Vision-Based RL Without Data Augmentation**  
Batra & Sukhatme, ICML 2024 — arXiv:2410.07441

---

## Problem

Vision-based RL agents fail on out-of-distribution (OOD) observations — different backgrounds, lighting, colors — even when the underlying task dynamics are identical. The dominant fix is data augmentation (DrQ, SVEA), but this scales poorly with the number of task variations and can destabilize training.

**ALDA's claim:** if you learn a properly disentangled latent representation and pair it with associative memory, you can remap OOD observations back to in-distribution ones at test time — zero-shot, no augmentation needed.

---

## Core Intuition

From neuroscience: the hippocampus stores memories by decomposing sensory input into independent factors. When a partially corrupted input is seen, pattern completion retrieves the closest stored memory.

Translated to RL:
- **Disentangle** the observation into independent latent variables (task-relevant vs. task-irrelevant)
- **Store** in-distribution latent codes in an associative memory during training
- At test time on OOD input: **retrieve** the nearest stored code for each latent dimension independently → remapped latent is back in-distribution → policy acts normally

Key insight: because latents are disentangled, you can remap a task-irrelevant factor (e.g., background color) without touching the task-relevant ones (e.g., agent pose).

---

## Components

### 1. QLAE — Quantized Latent AutoEncoder

QLAE (Hsu et al. 2023) is the disentanglement backbone. It learns:
- Encoder `f_θ: O → R^{n_z}` mapping an observation to `n_z` continuous scalar codes
- Decoder `g_φ: R^{n_z} → O` reconstructing the observation
- `n_z` scalar codebooks `V = V_1 × ... × V_{n_z}`, each with `|V_j|` discrete values

Each output dimension of the encoder is quantized independently to the nearest codebook entry:

```
z_{d_j} = argmin_{v ∈ V_j} |f_θ(x)_j - v|,   j = 1, ..., n_z
```

This gives a discrete latent `z_d ∈ Z`. Unlike VQ-VAE which uses a single shared codebook, QLAE uses one scalar codebook per dimension — this is what enforces disentanglement.

**QLAE losses** (straight-through gradient estimator):
```
L_quantize = || StopGradient(f_θ(x)) - z_d ||²
L_commit   = || f_θ(x) - StopGradient(z_d) ||²
```

**Temporal stacking for RL:** Since single frames are ambiguous, `k` consecutive RGB frames are stacked → input shape `R^{Bk×C×H×W}`. The encoder processes each frame individually, producing `Bk` latent vectors, which are reshaped into a 1D state vector `z ∈ R^{Bk×n_z}` used by the actor/critic.

**Dimensionality of z_d:** Set to match the size of the proprioceptive state space of the task. Paper uses `|z_d| = 12` across all reported tasks.

---

### 2. Associative Memory (Modern Hopfield Network)

The naive approach (feed quantized latents directly to a Hopfield network) doesn't work well in practice because QLAE's latent dynamics already *are* a Hopfield network implicitly.

Instead, the association is done directly on the codebook entries using a **Softmax retrieval**:

```
z_{d_s} = Softmax(-β · L1(f_θ(o), V)) ⊙ V          (Eq. 6)
```

Where:
- `V` = matrix of all stored codebook values (the memory)
- `f_θ(o)` = encoder output (continuous, pre-quantization)
- `L1(·, ·)` = L1 distance between encoder output and each codebook entry
- `β` = temperature (large β → hard nearest-neighbor; used large in practice)
- `⊙` = element-wise weighting / soft retrieval

**Interpretation:** This is a soft argmin over codebook entries. At high β it recovers hard quantization (Eq. 3). The associative memory mechanism is the codebook itself — no separate memory module needed. The QLAE codebook stores in-distribution values; OOD inputs get soft-mapped to the nearest in-distribution code per dimension.

---

## Training Objective

ALDA adds a consistency loss to standard SAC. The full objective:

```
J(ALDA) = E_{o ~ D} [
    || f_θ(o) - StopGradient(Softmax(-β·L1(f_θ(o), V)) ⊙ V) ||²   ← association loss
    + log g_φ(o | z_d)                                               ← reconstruction loss
    + λ_θ ||θ||²  + λ_φ ||φ||²                                       ← weight decay
]
```

**What this does:** the encoder is trained to produce outputs that are close to their own nearest codebook entry (the association target). This keeps the encoder outputs within the codebook's support, which is what makes OOD remapping work at test time.

Note: `L_quantize` is omitted (not optimized toward the codebook) — the codebook is only optimized implicitly via the encoder consistency. The paper empirically finds this keeps encoder outputs close to codebook values without the instability of quantize loss.

Weight decay `λ_θ = λ_φ = 0.1` (strong). SAC losses are added on top of this.

---

## Full Algorithm (Pseudocode)

```
Initialize:
  encoder f_θ (CNN), decoder g_φ (CNN)
  QLAE codebooks V = {V_1, ..., V_{n_z}}
  SAC actor π, critics Q1/Q2, target critics Q1'/Q2'
  replay buffer D
  log_alpha (entropy temperature, auto-tuned)

Training loop (each env step):
  1. Collect transition:
       stack k frames → o_t
       z_t = f_θ(o_t)                         # encode
       z_d  = quantize(z_t, V)                # nearest codebook entry per dim
       z    = reshape(z_d, Bk × n_z → 1D)    # temporal state vector
       a_t ~ π(·|z)                           # actor samples action
       o_{t+1}, r_t, done_t = env.step(a_t)
       D.add(o_t, a_t, r_t, o_{t+1}, done_t)

  2. Sample batch from D

  3. Compute ALDA auxiliary loss:
       z_enc = f_θ(o_batch)
       z_assoc = StopGradient(Softmax(-β·L1(z_enc, V)) ⊙ V)
       L_assoc = || z_enc - z_assoc ||²
       L_recon = -log g_φ(o_batch | z_d_batch)
       L_aux   = L_assoc + L_recon + weight_decay

  4. Compute SAC losses (operating on z = reshape(z_d)):
       - Critic loss: soft Bellman residual (twin Q, target networks)
       - Actor loss:  maximize E[Q(z, π(z)) - α log π(z)]
       - Alpha loss:  auto-tune entropy temperature

  5. Update all parameters:
       - θ, φ ← L_aux  (encoder + decoder)
       - Q1, Q2 ← critic loss
       - π ← actor loss  (every policy_frequency steps)
       - log_alpha ← alpha loss
       - Q1', Q2' ← polyak average (τ = 0.005)
```

---

## Key Hyperparameters

| Parameter | Value | Notes |
|---|---|---|
| `n_z` (latent dim) | 12 | Matched to proprioceptive state size |
| `β` (Hopfield temperature) | large (see appendix A.4) | Controls hardness of retrieval |
| `λ_θ`, `λ_φ` (weight decay) | 0.1 | Strong regularization on encoder/decoder |
| `k` (stacked frames) | 3 | Standard for DMC pixel tasks |
| Frame size | 84×84 | Standard DMC setting |
| SAC: `τ` | 0.005 | Polyak averaging coefficient |
| SAC: `γ` | 0.99 | Discount factor |
| SAC: batch size | 256 | — |
| SAC: buffer size | 1M | — |

---

## Experimental Setup

**Training:** DeepMind Control Suite (DMC) — proprioceptive tasks converted to pixel observations. Tasks evaluated: Cartpole Balance, Walker Walk, Ball-in-Cup Catch, Finger Spin.

**Evaluation (OOD):**
- **Color hard** — agent/background colors randomized to extreme RGB values
- **Distracting CS** — random video overlaid as background (DAVIS 2017 dataset)

**Baselines:**
- **DARLA** — only other method attempting disentangled representation for zero-shot generalization in vision RL
- **SAC+AE** — SAC with a deterministic autoencoder auxiliary objective
- **RePo** — task-centric latent representation immune to background distractors
- **SVEA** — augmentation-based (random overlay from Places dataset, 1.8M images)

---

## Results Summary

- ALDA matches or beats all non-SVEA baselines on both OOD benchmarks
- ALDA can outperform SVEA when SVEA is trained with augmentations that don't include the specific test-time distractor type
- SVEA still wins on DistractingCS when trained with matching augmentations (it sees 1.8M diverse backgrounds; ALDA sees none)
- Latent traversals (Fig. 6) confirm disentanglement: individual latents encode interpretable factors (e.g., agent orientation, background color) independently

---

## Why Data Augmentation = Weak Disentanglement

The paper proves (Section 3) that data augmentation methods implicitly learn a *weakly* disentangled latent — the latent is partitioned into task-relevant and task-irrelevant subspaces, but not fully factorized into individual independent components. ALDA achieves stronger (full) disentanglement per dimension, which is why it can remap individual latent dimensions independently at test time.
