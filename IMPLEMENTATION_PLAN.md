# ALDA Implementation Plan

---

## 1. Paper Pipeline — What Is Actually Happening

### 1.1 The three-component view

```
Pixel obs (84×84×3 per frame)
        │
        ▼  encode each frame separately
   PixelEncoder f_θ                   ← existing encoder.py, in_channels=3, latent_dim=12
        │ z_enc: (B, n_z=12) per frame
        ▼
   QLAE Codebook V: (n_z=12, K=10)   ← new: qlae.py
        │  per-dim soft argmin (Eq 6)
        │  z_s = Softmax(-β·|z_enc - V|) @ V
        │
        ├── training:  L_assoc = ||z_enc - sg(z_s)||²  (encoder toward codebook)
        │              L_recon = MSE(decoder(z_s), obs)
        │              L_wd    = λ||θ||² + λ||φ||²
        │
        └── acting/critic:  z = concat(z_s_1, z_s_2, z_s_3) → (B, k·n_z=36)
                │
                ▼
           SAC Actor / Twin Critics                   ← existing, just change input dim
```

### 1.2 Key insight — the codebook IS the associative memory

There is **no separate memory module**. The QLAE codebook V stores K=10 "prototype" values
for each of the n_z=12 latent dimensions. When an OOD frame arrives:
- Its encoder output z_enc may fall outside the in-distribution support
- Eq 6 soft-maps it to the nearest prototype per dimension
- Because dimensions are independent, only the corrupted dimensions (e.g. color) get remapped
- Task-relevant dimensions (e.g. pose) are already in-distribution → map to themselves

### 1.3 Crucially: what is NOT in the paper that must be decided

| Question | Paper says | Implementation decision |
|---|---|---|
| How is codebook updated? | L_quantize omitted; implicit via L_assoc | Codebook = nn.Parameter, updated by AdamW with encoder |
| Codebook initialization | Not specified | Uniform(-1, 1) per dim, K points |
| Hard vs soft quantization during collection | Not explicit | Use soft (Eq 6) for everything — avoids hard argmin in rollout |
| Does critic see z_enc or z_s? | z_d (quantized) in pseudocode | Use z_s (soft retrieval) — differentiable, same at high β |
| Does encoder get updated by critic loss? | Yes (same as SAC+AE) | Yes — enc_opt zero_grad before critic backward |

---

## 2. Pseudocode: ALDA Training Step

```
# Shapes: B=batch_size=256, k=3, n_z=12, K=10, C=3 (RGB), H=W=84

# ─── ENCODE (shared function used in collect + train) ───────────────────────
def encode_frames(obs_stacked, encoder, codebook, beta):
    # obs_stacked: (B, k*C, H, W) = (B, 9, 84, 84)
    B = obs_stacked.shape[0]
    frames = obs_stacked.view(B * k, C, H, W)            # (B*3, 3, 84, 84)
    z_enc = encoder(frames)                               # (B*3, 12)
    z_s = associate(z_enc, codebook, beta)               # (B*3, 12)  — soft retrieval
    z = z_s.view(B, k * n_z)                             # (B, 36)
    return z, z_enc                                       # z for policy; z_enc for loss

def associate(z_enc, codebook, beta):
    # z_enc:    (N, n_z)
    # codebook: (n_z, K)
    l1 = (z_enc.unsqueeze(-1) - codebook.unsqueeze(0)).abs()   # (N, n_z, K)
    w  = torch.softmax(-beta * l1, dim=-1)                      # (N, n_z, K)
    return (w * codebook.unsqueeze(0)).sum(-1)                  # (N, n_z)

# ─── COLLECT ────────────────────────────────────────────────────────────────
z, _ = encode_frames(obs, encoder, codebook, beta)        # (1, 36)
action, _, _ = actor.get_action(z)
obs_next, reward, term, trunc, info = env.step(action)
buffer.add(obs, action, reward, obs_next, done)
obs = obs_next

# ─── SAMPLE BATCH ───────────────────────────────────────────────────────────
obs_b, next_b, act_b, rew_b, done_b = buffer.sample(B)   # obs_b: (B, 9, 84, 84)

# ─── STEP 1: Encoder + Critic update ────────────────────────────────────────
with torch.no_grad():
    z_next, _ = encode_frames(next_b, encoder, codebook, beta)
    na, log_pi, _ = actor.get_action(z_next)
    q_next = min(qf1_t(z_next, na), qf2_t(z_next, na)) - alpha * log_pi
    q_tgt  = rew_b + (1 - done_b) * gamma * q_next

z_b, z_enc_b = encode_frames(obs_b, encoder, codebook, beta)   # z_b detaches through codebook
qf_loss = mse(qf1(z_b, act_b), q_tgt) + mse(qf2(z_b, act_b), q_tgt)

enc_opt.zero_grad(); qf_opt.zero_grad()
qf_loss.backward()
qf_opt.step(); enc_opt.step()

# ─── STEP 2: Auxiliary losses (encoder + decoder + codebook) ────────────────
z_b2, z_enc_b2 = encode_frames(obs_b, encoder, codebook, beta)
z_s_sg = associate(z_enc_b2, codebook, beta).detach()           # target: sg(z_s)
L_assoc = mse(z_enc_b2.view(B, k, n_z), z_s_sg.view(B, k, n_z))

# Decode from the full (B*k) batch → single-frame reconstructions
recon = decoder(z_s_sg)                                          # (B*k, 3, 84, 84)
obs_frames = obs_b.view(B * k, C, H, W)
L_recon = mse(recon, obs_frames)

L_wd = lambda_enc * sum(p.pow(2).mean() for p in encoder.parameters()) \
     + lambda_dec * sum(p.pow(2).mean() for p in decoder.parameters())

L_aux = L_assoc + L_recon + L_wd

enc_opt.zero_grad(); dec_opt.zero_grad(); cb_opt.zero_grad()
L_aux.backward()
enc_opt.step(); dec_opt.step(); cb_opt.step()

# ─── STEP 3: Actor + alpha (encoder and codebook detached) ──────────────────
if step % policy_frequency == 0:
    z_det, _ = encode_frames(obs_b, encoder.detach_mode(), codebook, beta)
    # NOTE: use detach=True in encoder forward, same as SAC+AE
    pi, log_pi, _ = actor.get_action(z_det)
    actor_loss = (alpha * log_pi - min(qf1(z_det, pi), qf2(z_det, pi))).mean()
    actor_opt.zero_grad(); actor_loss.backward(); actor_opt.step()
    # alpha update same as before

# ─── STEP 4: Target network polyak ──────────────────────────────────────────
for p, tp in zip(qf1.params, qf1_t.params):
    tp.data = tau * p.data + (1-tau) * tp.data
```

---

## 3. Files to Write

### 3.1 `alda/qlae.py` — QLAE module

**Responsibility:** holds codebook V, implements `associate()`, exposes the straight-through
quantized output and the association loss.

```python
class QLAE(nn.Module):
    """
    Quantized Latent AutoEncoder codebook.

    Holds V: (n_z, K) learnable scalar prototypes.
    Two modes:
      - soft (training): z_s = Softmax(-β·L1) @ V   (differentiable, used for everything)
      - hard (legacy):   z_d = V[argmin L1]          (straight-through; not used in ALDA)

    In ALDA, soft mode is used everywhere (Eq. 6). Hard mode left for ablation.
    """
    def __init__(self, n_z: int = 12, K: int = 10, beta: float = 10.0):
        super().__init__()
        self.n_z  = n_z
        self.K    = K
        self.beta = beta
        # Codebook: one row per latent dim, K prototype values each
        # Init uniform(-1, 1) — encoder output range is roughly [-1,1] (tanh encoder)
        self.codebook = nn.Parameter(
            torch.zeros(n_z, K).uniform_(-1.0, 1.0)
        )

    def associate(self, z_enc: torch.Tensor) -> torch.Tensor:
        """
        Soft nearest-neighbour retrieval (Eq. 6).
        z_enc: (N, n_z)  →  z_s: (N, n_z)
        """
        # l1: (N, n_z, K)
        l1 = (z_enc.unsqueeze(-1) - self.codebook.unsqueeze(0)).abs()
        w  = torch.softmax(-self.beta * l1, dim=-1)   # (N, n_z, K)
        return (w * self.codebook.unsqueeze(0)).sum(-1) # (N, n_z)

    def association_loss(self, z_enc: torch.Tensor) -> torch.Tensor:
        """
        L_assoc = ||z_enc - sg(z_s)||²  — pushes encoder toward codebook.
        z_enc: (N, n_z)
        """
        z_s = self.associate(z_enc).detach()
        return F.mse_loss(z_enc, z_s)

    def codebook_usage(self, z_enc: torch.Tensor) -> float:
        """Fraction of codebook slots that are nearest-neighbour for some z_enc. Diagnostic."""
        with torch.no_grad():
            l1 = (z_enc.unsqueeze(-1) - self.codebook.unsqueeze(0)).abs()
            used = l1.argmin(-1).unique().numel()
        return used / (self.n_z * self.K)

    def forward(self, z_enc: torch.Tensor) -> torch.Tensor:
        """Returns z_s (soft-quantized). Used for actor/critic input."""
        return self.associate(z_enc)
```

No `L_quantize` (codebook is NOT pulled toward encoder — paper omits this intentionally).

### 3.2 `alda/memory.py` — thin wrapper (mostly for eval.py OOD remapping)

```python
"""
At test time (OOD), call remap() to correct z_enc before passing to policy.
Training: QLAE.association_loss() is used instead — this file is for clarity/eval.
"""

import torch

def remap(z_enc: torch.Tensor, qlae) -> torch.Tensor:
    """
    OOD correction: replace z_enc with its soft nearest-neighbour in codebook.
    z_enc: (N, n_z) — raw encoder output on OOD frame
    Returns: z_s: (N, n_z) — in-distribution approximation
    """
    with torch.no_grad():
        return qlae.associate(z_enc)
```

This is literally 6 lines. The heavy lifting is in QLAE.

### 3.3 Modified `alda/encoder.py` (small change)

The existing encoder already supports `in_channels` as a constructor arg. For ALDA:
- `PixelEncoder(in_channels=3, latent_dim=12)` — single frame, smaller latent

No code change needed; just instantiate differently.

The `detach` kwarg in `forward()` already exists — used for actor updates. ✓

### 3.4 `train_alda.py` — main training loop

**Diffs vs `train_sac_pixel.py`:**

| Aspect | SAC+AE | ALDA |
|---|---|---|
| Encoder input | (B, 9, 84, 84) stacked | (B*k, 3, 84, 84) per-frame |
| Latent dim | 50 | 12 |
| Policy input dim | 50 | k * 12 = 36 |
| Codebook | none | QLAE(n_z=12, K=10) with own optimizer cb_opt |
| Aux losses | L_recon only | L_assoc + L_recon + L_wd |
| Optimizers | enc_opt, dec_opt, qf_opt, actor_opt | + cb_opt (AdamW on codebook) |
| Weight decay | none | AdamW with weight_decay=0.1 for enc/dec |
| At eval time | encoder(obs) → 50-dim | encode_frames(obs) + qlae.associate() → 36-dim |

**New helper added to train_alda.py:**
```python
def encode_frames(obs_b, encoder, qlae, detach=False):
    """
    obs_b:   (B, k*C, H, W) = (B, 9, 84, 84)
    returns: z: (B, k*n_z), z_enc_flat: (B*k, n_z)
    """
    B = obs_b.shape[0]
    frames = obs_b.view(B * k, C, H, W)          # (B*3, 3, 84, 84)
    z_enc  = encoder(frames, detach=detach)        # (B*3, n_z)
    z_s    = qlae(z_enc)                           # (B*3, n_z)  soft retrieval
    z      = z_s.view(B, k * n_z)                 # (B, 36)
    return z, z_enc
```

**Optimizers:**
```python
enc_opt   = optim.AdamW(encoder.parameters(), lr=recon_lr, weight_decay=lambda_enc)
dec_opt   = optim.AdamW(decoder.parameters(), lr=recon_lr, weight_decay=lambda_dec)
cb_opt    = optim.Adam(qlae.parameters(),   lr=recon_lr)   # codebook; no extra WD
qf_opt    = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=q_lr)
actor_opt = optim.Adam(actor.parameters(), lr=policy_lr)
```

Using AdamW for encoder+decoder because the paper's L_wd = λ||θ||² is exactly L2 weight decay
— AdamW implements this correctly (separate from gradient momentum).

### 3.5 Changes to `eval.py`

Add `--alda` flag. When set:
1. Load `qlae` state dict from checkpoint (need to save it in `train_alda.py`)
2. In rollout, call `encode_frames(obs, encoder, qlae)` instead of `encoder(obs)`

The `evaluate_ood()` inline function in the training loop also needs to be updated.

### 3.6 `train_alda.py` checkpoint format

```python
torch.save({
    "step": step,
    "args": vars(args),
    "encoder": encoder.state_dict(),
    "qlae":    qlae.state_dict(),       # ← new vs SAC+AE
    "decoder": decoder.state_dict(),    # ← save this too for ALDA (used in eval)
    "actor":   actor.state_dict(),
    "action_scale": actor.action_scale.cpu(),
    "action_bias":  actor.action_bias.cpu(),
}, ckpt_path)
```

---

## 4. Optimizer / Gradient Flow Map

```
           ┌────────────────────────────────────────────────────┐
           │  obs_b (B, 9, 84, 84)                              │
           └─────────┬──────────────────────────────────────────┘
                     │  view → (B*k, 3, 84, 84)
                     ▼
              PixelEncoder f_θ        ← enc_opt (AdamW, lr=1e-3, wd=0.1)
                     │ z_enc: (B*k, 12)
                     │
              ┌──────┴───────────────────┐
              │                          │
              ▼  STEP 1 (critic)         ▼  STEP 2 (aux)
         QLAE.associate()          QLAE.associate()
              │ z_s (B*k,12)            │ z_s_sg = z_s.detach()
              │ view → (B,36)           │
              ▼                         ├── L_assoc = mse(z_enc, z_s_sg)
         qf1(z_s, act_b) ──────────    │              ← gradients into f_θ and codebook
         qf2(z_s, act_b)    qf_loss    │
              │                         ├── L_recon = mse(decoder(z_s_sg), obs_frames)
         qf_loss.backward()             │              ← gradients into decoder
              │                         │
         enc_opt.step()  ◄──── grads   ▼
         qf_opt.step()            enc_opt.step()
                                  dec_opt.step()
                                  cb_opt.step()

              STEP 3 (actor):
         z_det, _ = encode_frames(..., detach=True)   ← encoder output .detach()
         actor_loss.backward()
         actor_opt.step()
         (no gradient into encoder or codebook)
```

**Three separate encoder gradient contributions per step:**
1. Critic loss → encoder (step 1)
2. Association loss → encoder + codebook (step 2)
3. Reconstruction loss → decoder only (z_enc already detached via z_s_sg) (step 2)

---

## 5. Potential Issues and Mitigations

| Issue | Symptom | Fix |
|---|---|---|
| Codebook collapse | `codebook_usage < 0.3` | Reduce β initially (start at 1.0, anneal to 10); reinit collapsed entries |
| Encoder outputs diverge from codebook | `L_assoc` doesn't decrease | Increase `lambda_enc` weight on L_assoc; check AdamW wd |
| Gradient explosion in step 1 | NaN qf_loss | Add gradient clip (max_norm=10) to enc_opt and qf_opt |
| Policy doesn't learn (36-dim input unfamiliar) | train_return stays near 0 | Verify `encode_frames` reshape is correct; print z shape at step 0 |
| DistractingCS at eval crashes | Exception mid-eval | Wrap in try/except in evaluate_ood; log NaN |
| AssocLoss interferes with SAC | train_return degrades vs SAC+AE | Ablate: first train with L_assoc_weight=0 (= SAC+AE with n_z=12), then add it |

---

## 6. Build Order

```
Step 1  alda/qlae.py
        ├── Write QLAE class (codebook, associate, association_loss, codebook_usage)
        └── Unit test: assert output shape, gradient flows, codebook_usage > 0 after 100 random inputs

Step 2  train_alda.py
        ├── Copy train_sac_pixel.py as base
        ├── Add encode_frames() helper
        ├── Replace enc init: PixelEncoder(in_channels=3, latent_dim=12)
        ├── Add QLAE init + cb_opt
        ├── Replace enc_opt with AdamW
        ├── Replace critic/actor forward calls to use encode_frames()
        ├── Add Step 2 auxiliary losses
        ├── Update evaluate_ood() to use encode_frames()
        └── Smoke test: cartpole-balance, 2000 steps, --track

Step 3  eval.py update
        ├── Load qlae from checkpoint
        ├── Use encode_frames() in rollout
        └── Test with smoke-test checkpoint

Step 4  alda/memory.py
        └── remap() wrapper (for clarity; used in eval.py)
```

Total new code: ~250 lines (`qlae.py` ~80, `train_alda.py` ~420 lines with diffs from SAC+AE,
`memory.py` ~20, eval.py update ~30).

---

## 7. Quick Reference: Shape Table

| Tensor | SAC+AE | ALDA |
|---|---|---|
| obs stored in buffer | (B, 9, 84, 84) | (B, 9, 84, 84) — same |
| encoder input | (B, 9, 84, 84) | (B*k=B*3, 3, 84, 84) |
| encoder output z_enc | (B, 50) | (B*3, 12) |
| after quantization z_s | — | (B*3, 12) |
| policy input z | (B, 50) | (B, 36) = reshape(B*3, 12) |
| decoder input | (B, 50) | (B*3, 12) — per-frame |
| decoder output | (B, 9, 84, 84) | (B*3, 3, 84, 84) — per-frame |
| recon target | obs_b | frames_b = obs_b.view(B*3, 3, 84, 84) |
| codebook | — | (12, 10) |
