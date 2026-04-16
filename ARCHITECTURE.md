# Codebase Architecture

---

## File Status

### Active pipeline (everything needed to run experiments)

| File | Role |
|---|---|
| `train_alda.py` | Primary training script — full ALDA agent |
| `train_sac_pixel.py` | Baseline training script — SAC+AE for comparison |
| `eval.py` | Post-hoc OOD evaluation of saved checkpoints |
| `envs.py` | DMC + DistractingCS wrappers — used by all three above |
| `alda/encoder.py` | CNN encoder (shared by both algos, different instantiation) |
| `alda/decoder.py` | CNN decoder (shared by both algos) |
| `alda/qlae.py` | QLAE codebook — ALDA only |
| `alda/memory.py` | `remap()` wrapper around QLAE — used in eval |
| `alda/replay_buffer.py` | uint8 pixel replay buffer — shared |
| `pyproject.toml` + `uv.lock` | dependency management |
| `scripts/test_quick.sh` | pre-flight end-to-end test (~2 min) |
| `scripts/run_cloud.sh` | cloud experiment launcher |
| `scripts/setup_kaggle.sh` | Kaggle environment setup |
| `WANDB_API_KEY.md` | credentials — **do not commit to public git** |

---

## Supported Environments

### Training — DeepMind Control Suite (DMC)

Any `dm_control` task registered via `shimmy`. Tested tasks:

| `--task` | Domain | Notes |
|---|---|---|
| `walker-walk` | Walker | Main paper benchmark |
| `cartpole-balance` | Cartpole | Easier, fast to smoke-test |
| `ball_in_cup-catch` | Ball-in-cup | Also in paper |
| `finger-spin` | Finger | Also in paper |
| `cheetah-run` | Cheetah | Harder |

Registered as `gym.make("dm_control/<task>-v0")`. Requires `shimmy` and `mujoco`.

### Evaluation — Distracting Control Suite (DistractingCS)

| Distraction | Difficulties | Extra data? | Works on cloud? |
|---|---|---|---|
| `color` | easy / medium / hard | **No** — HSV color jitter | **Yes** — fully self-contained |
| `camera` | easy / medium / hard | **No** — camera pose perturbation | **Yes** |
| `background` | easy / medium / hard | **Yes** — DAVIS 2017 video (~6 GB) | Only if dataset uploaded |

**OOD eval during training** (both training scripts) runs **color-hard only** — no external
dataset, works headless with `MUJOCO_GL=egl`. It fires every `eval_frequency` steps inside
the training loop. Background distraction is only triggered by `eval.py` when
`--background-dataset-path` is explicitly provided.

`make_dcs()` takes the task underscore-separated (e.g. `walker_walk`).
`evaluate_ood()` converts automatically: `args.task.replace("-", "_")`.

---

## Agent Structures

### SAC+AE (`train_sac_pixel.py`)

```
Components:
  encoder   PixelEncoder(in_channels=9, latent_dim=50)  ← stacked (9,84,84) input
  decoder   PixelDecoder(out_channels=9, latent_dim=50)
  actor     Actor(policy_dim=50, act_dim)
  qf1/qf2   SoftQNetwork(policy_dim=50, act_dim)
  qf1_t/qf2_t  — polyak-averaged target critics

Optimisers:
  enc_opt   Adam(encoder,  lr=1e-3)
  dec_opt   Adam(decoder,  lr=1e-3)
  qf_opt    Adam(qf1+qf2,  lr=1e-3)
  actor_opt Adam(actor,    lr=3e-4)
  alpha_opt Adam([log_α],  lr=1e-3)   — autotune only
```

### ALDA (`train_alda.py`)

```
Components:
  encoder   PixelEncoder(in_channels=3, latent_dim=12)  ← single frame (3,84,84) input
  decoder   PixelDecoder(out_channels=3, latent_dim=12)
  qlae      QLAE(n_z=12, K=10, beta=10)                 ← codebook (12,10)
  actor     Actor(policy_dim=36, act_dim)                ← 36 = 3 frames × 12 codes
  qf1/qf2   SoftQNetwork(policy_dim=36, act_dim)
  qf1_t/qf2_t

Optimisers:
  enc_opt   AdamW(encoder, lr=1e-3, weight_decay=0.1)   ← AdamW implements paper's λ_θ
  dec_opt   AdamW(decoder, lr=1e-3, weight_decay=0.1)
  cb_opt    Adam(qlae,     lr=1e-3)                      ← plain Adam; prototypes not penalised to 0
  qf_opt    Adam(qf1+qf2,  lr=1e-3)
  actor_opt Adam(actor,    lr=3e-4)
  alpha_opt Adam([log_α],  lr=1e-3)
```

Key difference: ALDA encodes each of the k=3 frames independently → QLAE retrieves
a 12-dim code per frame → concatenate to 36-dim policy input.
SAC+AE encodes all 3 frames jointly as a single (9, 84, 84) tensor → 50-dim policy input.

---

## Training Loop Structure (both scripts)

```
for step in range(total_timesteps):

  ── Collect ──────────────────────────────────────────────────────────
  if step < learning_starts:
      action = random
  else:
      encoder.eval()
      z = encode(obs)           # encoder + qlae for ALDA; encoder only for SAC+AE
      action = actor.get_action(z)   # stochastic sample
      encoder.train()
  next_obs, reward, term, trunc, info = env.step(action)

  # Gymnasium 1.x episode logging:
  if "episode" in infos:
      ep_r = infos["episode"]["r"][idx]   # NOT infos["final_info"]

  # Terminal obs for truncation:
  if "final_observation" in infos: ...    # guarded — not always present

  rb.add(obs, next_obs, action, reward, termination)
  # Note: termination (not truncation) as done flag

  if step < learning_starts: continue

  ── Train ────────────────────────────────────────────────────────────
  obs_b, next_b, act_b, rew_b, done_b = rb.sample(batch_size)

  Step 1  Critic + encoder
  Step 2  Auxiliary (recon; ALDA also adds assoc_loss + weight decay)
  Step 3  Actor + alpha  (every policy_frequency=2 steps)
  Step 4  Target network polyak  (every target_frequency=1 steps)

  ── Log every 1 000 steps ────────────────────────────────────────────
  ── OOD eval (color-hard) + checkpoint every eval_frequency steps ───
```

---

## Backprop Flow

### SAC+AE

```
Step 1 — Critic + encoder
  obs_b → encoder → z(B,50) → qf1/qf2(z, act_b) → qf_loss
  qf_loss.backward()
  enc_opt.step()    ← encoder updated by critic gradient
  qf_opt.step()

Step 2 — Reconstruction
  obs_b → encoder → z(B,50) → decoder → recon
  recon_loss = mse(recon, obs_b)
  recon_loss.backward()
  enc_opt.step()    ← encoder updated again by recon gradient
  dec_opt.step()

Step 3 — Actor + alpha (encoder blocked)
  z_det = encoder(obs_b, detach=True)   ← gradient STOPS at encoder output
  actor_loss = (α·log_π − min(Q1,Q2)).mean()
  actor_opt.step()                      ← encoder NOT touched
  alpha_opt.step()
```

### ALDA

```
Step 1 — Critic + encoder
  obs_b → reshape(B*3, 3, 84, 84) → encoder → z_enc(B*3, 12)
         → qlae.associate() → z_s(B*3, 12) → reshape(B, 36)
         → qf1/qf2(z, act_b) → qf_loss
  qf_loss.backward()
  enc_opt.step()    ← encoder updated; codebook gradient accumulated but NOT applied
  qf_opt.step()     ← cb_opt NOT stepped here

Step 2 — Auxiliary losses
  cb_opt.zero_grad()               ← clears step-1 codebook gradient (discarded)
  z_enc = encoder(frames)
  z_s   = qlae.associate(z_enc)

  L_assoc = mse(z_enc, z_s.detach())   ← encoder toward codebook; StopGrad on codebook
  L_recon = mse(decoder(z_s), frames)  ← z_s NOT detached → codebook gets gradient here

  (L_assoc + L_recon).backward()
  enc_opt.step()    ← encoder: from L_assoc + L_recon + AdamW weight decay
  dec_opt.step()    ← decoder: from L_recon + AdamW weight decay
  cb_opt.step()     ← codebook: from L_recon ONLY (L_assoc has StopGrad on codebook)

Step 3 — Actor + alpha (encoder + codebook blocked)
  z_det = encode_frames(obs_b, encoder, qlae, detach_encoder=True)
  actor_loss → actor_opt    ← no gradient into encoder or codebook
  alpha_opt.step()
```

---

## Key Symbols and Duplicates

### Shared network classes (duplicated across files)

| Symbol | `train_sac_pixel.py` | `train_alda.py` | `eval.py` | Notes |
|---|---|---|---|---|
| `Actor` | ✓ (`latent_dim`) | ✓ (`policy_dim`) | ✓ (`policy_dim`) | Identical logic; only constructor arg name differs |
| `SoftQNetwork` | ✓ | ✓ | — | Identical |
| `_maybe_set_wandb_key` | ✓ | ✓ | ✓ | Identical |
| `make_train_env` | ✓ | ✓ | — | Identical |
| `evaluate_ood` | ✓ SAC+AE version | ✓ ALDA version | replaced by `evaluate_all` | Differ in encode call |
| `Args` dataclass | ✓ | ✓ (extended) | — | ~80% overlapping fields |
| Episode stat handling | ✓ | ✓ | — | Identical gymnasium 1.x compat code |
| Target network polyak | ✓ | ✓ | — | Identical |

**Practical consequence:** changing `Actor` (e.g. adding layer norm) requires editing 3 files.
Consolidating into `alda/nets.py` is the right fix before the paper-writing phase.

### Unique to each file

| Symbol | File | Description |
|---|---|---|
| `encode_frames()` | `train_alda.py` | Reshape stacked obs → per-frame encoding → QLAE → concat |
| `QLAE` | `alda/qlae.py` | Codebook, `associate()`, `association_loss()`, `codebook_usage()` |
| `remap()` | `alda/memory.py` | Thin `@no_grad` wrapper around `qlae.associate()` for eval |
| `encode_sac_ae()` / `encode_alda()` | `eval.py` | Dispatch functions; auto-selected from checkpoint keys |
| `evaluate_all()` | `eval.py` | Sweeps all distraction × difficulty combos, returns result list |
| `PixelReplayBuffer` | `alda/replay_buffer.py` | uint8 storage, `_f2u`/`_u2ft` conversion helpers |
| `PixelEncoder` | `alda/encoder.py` | 4-layer CNN, LayerNorm, Tanh; `detach` kwarg for actor updates |
| `PixelDecoder` | `alda/decoder.py` | 4 ConvTranspose layers, `output_padding=1` on last layer |

---

## Optimizer Structure

### SAC+AE — 4 optimizers

| Optimizer | Parameters | Updated in |
|---|---|---|
| `enc_opt` Adam | encoder | Step 1 (critic) AND Step 2 (recon) |
| `dec_opt` Adam | decoder | Step 2 (recon) |
| `qf_opt` Adam | qf1, qf2 | Step 1 (critic) |
| `actor_opt` Adam | actor | Step 3 (actor) |

### ALDA — 5 optimizers

| Optimizer | Parameters | Updated in | Note |
|---|---|---|---|
| `enc_opt` AdamW wd=0.1 | encoder | Step 1 (critic) AND Step 2 (aux) | AdamW WD applied twice per global step |
| `dec_opt` AdamW wd=0.1 | decoder | Step 2 (aux) | |
| `cb_opt` Adam | qlae codebook | Step 2 (aux) only | Step-1 codebook grad discarded via `cb_opt.zero_grad()` |
| `qf_opt` Adam | qf1, qf2 | Step 1 (critic) | |
| `actor_opt` Adam | actor | Step 3 (actor) | |

---

## Checkpoint Formats

### SAC+AE
```python
{"step", "args", "encoder", "actor", "action_scale", "action_bias"}
```

### ALDA
```python
{"step", "args", "encoder", "qlae", "decoder", "actor", "action_scale", "action_bias"}
```

`eval.py` auto-detects algo from checkpoint: `"qlae" in ckpt → ALDA`, else `SAC+AE`.
