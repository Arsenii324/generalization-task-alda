"""
ALDA training — SAC + QLAE (per-dim codebooks) + Softmax associative memory.

Batra & Sukhatme, ICML 2024 — arXiv:2410.07441

Key differences vs train_sac_pixel.py (SAC+AE):
  - Encoder sees single frames (3, 84, 84), not the stacked (9, 84, 84) tensor.
    Each of the k=3 stacked frames is encoded separately; latent codes are
    concatenated → policy input dim = k × n_z = 3 × 12 = 36.
  - Latent dim n_z=12 (vs 50 in SAC+AE), matching the paper.
  - QLAE codebook: (n_z=12, K=10) learnable parameters.
  - 5 optimisers: enc (AdamW), dec (AdamW), cb (Adam), qf (Adam), actor (Adam).
    AdamW on enc/dec implements the paper's weight decay λ_θ, λ_φ = 0.1.
  - Aux losses: L_assoc (encoder → codebook) + L_recon (decoder, via z_s).
  - Codebook updated only by L_recon, not by critic or L_assoc (StopGrad per Eq. 7).

Architectural decisions (see also IMPLEMENTATION_PLAN.md):
  D1: Codebook gradient sources.
      L_assoc has StopGrad on z_s → gradient into encoder only, NOT codebook.
      L_recon uses z_s (not detached) as decoder input → gradient into decoder
      AND codebook AND encoder.  Critic loss does NOT update codebook (cb_opt
      not stepped in step 1; any accumulated grad is cleared by cb_opt.zero_grad
      at the start of step 2).

  D2: Soft retrieval everywhere (training + collection + eval).
      At β=10 on a well-trained encoder this is numerically identical to hard
      argmin, so there is no train/eval mismatch.

  D3: Weight decay applied twice per global step for encoder (once in step 1
      via enc_opt.step, once in step 2).  This is a minor over-regularisation
      accepted for simplicity; reducing lambda_enc by 2× is an easy ablation.

  D4: Target-network encoding uses the same QLAE codebook (no target codebook).
      The codebook evolves slowly and a target copy would add complexity with
      unclear benefit.

Run (smoke test, ≤2k steps):
    MUJOCO_GL=glfw uv run python train_alda.py \
        --task cartpole-balance --seed 1 \
        --total-timesteps 2000 --learning-starts 500 \
        --eval-frequency 1000 --n-eval-episodes 2 --track

Run (full, cloud):
    MUJOCO_GL=glfw uv run python train_alda.py \
        --task walker-walk --seed 1 --total-timesteps 500000 --track
"""

import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from torch.utils.tensorboard import SummaryWriter

import shimmy  # noqa: F401
from alda.encoder import PixelEncoder
from alda.decoder import PixelDecoder
from alda.qlae import QLAE
from alda.replay_buffer import PixelReplayBuffer
from envs import make_dmc, make_dcs


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class Args:
    # Experiment
    task: str = "walker-walk"
    """DMC task: walker-walk | cartpole-balance | ball_in_cup-catch | finger-spin"""
    seed: int = 1
    total_timesteps: int = 500_000

    # Logging
    track: bool = False
    wandb_project: str = "alda-sac-pixel"
    wandb_entity: str | None = None
    eval_frequency: int = 10_000
    """Steps between OOD evaluation episodes."""
    n_eval_episodes: int = 10
    """Episodes per OOD evaluation."""
    checkpoint_dir: str = "checkpoints"
    """Directory to save encoder+qlae+actor checkpoints."""

    # Encoder / decoder
    n_z: int = 12
    """Latent dimensions per frame (=12 across all tasks, paper Appendix A)."""
    frame_stack: int = 3
    image_size: int = 84
    recon_lr: float = 1e-3
    lambda_enc: float = 0.1
    """AdamW weight decay for encoder (= λ_θ in paper)."""
    lambda_dec: float = 0.1
    """AdamW weight decay for decoder (= λ_φ in paper)."""

    # QLAE / associative memory
    codebook_size: int = 10
    """Prototype values per latent dimension (=K in paper)."""
    beta: float = 10.0
    """Softmax temperature for associative retrieval (large → near argmin)."""

    # SAC
    buffer_size: int = 100_000
    gamma: float = 0.99
    tau: float = 0.005
    batch_size: int = 256
    learning_starts: int = 1_000
    policy_lr: float = 3e-4
    q_lr: float = 1e-3
    policy_frequency: int = 2
    target_frequency: int = 1
    autotune: bool = True
    alpha: float = 0.2

    # Hardware
    cuda: bool = True
    torch_deterministic: bool = True

    # Derived
    run_name: str = field(init=False)

    def __post_init__(self):
        self.run_name = f"alda__{self.task}__seed{self.seed}__{int(time.time())}"

    @property
    def policy_dim(self) -> int:
        """Input dimension to actor / critics: k frames × n_z codes each."""
        return self.frame_stack * self.n_z


# ---------------------------------------------------------------------------
# Networks (identical to SAC+AE; input dim changes from 50 → 36)
# ---------------------------------------------------------------------------

LOG_STD_MAX, LOG_STD_MIN = 2, -5


class Actor(nn.Module):
    def __init__(self, policy_dim: int, act_dim: int,
                 action_scale: torch.Tensor, action_bias: torch.Tensor):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(policy_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
        )
        self.fc_mean   = nn.Linear(256, act_dim)
        self.fc_logstd = nn.Linear(256, act_dim)
        self.register_buffer("action_scale", action_scale)
        self.register_buffer("action_bias",  action_bias)

    def forward(self, z: torch.Tensor):
        h       = self.net(z)
        mean    = self.fc_mean(h)
        log_std = torch.tanh(self.fc_logstd(h))
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
        return mean, log_std

    def get_action(self, z: torch.Tensor):
        mean, log_std = self(z)
        std    = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t    = normal.rsample()
        y_t    = torch.tanh(x_t)
        action   = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t) - torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean_act = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean_act

    @torch.no_grad()
    def get_mean_action(self, z: torch.Tensor) -> torch.Tensor:
        mean, _ = self(z)
        return torch.tanh(mean) * self.action_scale + self.action_bias


class SoftQNetwork(nn.Module):
    def __init__(self, policy_dim: int, act_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(policy_dim + act_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([z, a], dim=1))


# ---------------------------------------------------------------------------
# Encoding helper
# ---------------------------------------------------------------------------

C_PER_FRAME = 3  # RGB


def encode_frames(
    obs_stacked: torch.Tensor,
    encoder: PixelEncoder,
    qlae: QLAE,
    detach_encoder: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Encode a batch of frame-stacked observations through encoder + QLAE.

    Each frame is encoded independently; codes are concatenated to form
    the policy input (see paper temporal stacking, Section 3.2).

    Args:
        obs_stacked:    (B, k*C, H, W)  — stacked pixel obs, float32 in [-0.5, 0.5]
        encoder:        PixelEncoder with in_channels=C=3, latent_dim=n_z
        qlae:           QLAE module (holds codebook)
        detach_encoder: if True, stop gradient at encoder output (for actor updates)

    Returns:
        z:      (B, k*n_z)  — concatenated soft-retrieved codes; use for policy
        z_enc:  (B*k, n_z)  — raw encoder output before QLAE; use for L_assoc
    """
    B = obs_stacked.shape[0]
    k = obs_stacked.shape[1] // C_PER_FRAME           # should equal frame_stack
    frames = obs_stacked.view(B * k, C_PER_FRAME,
                              obs_stacked.shape[2],
                              obs_stacked.shape[3])    # (B*k, 3, H, W)
    z_enc = encoder(frames, detach=detach_encoder)     # (B*k, n_z)
    z_s   = qlae(z_enc)                               # (B*k, n_z)  soft retrieval
    z     = z_s.view(B, k * qlae.n_z)                 # (B, k*n_z)
    return z, z_enc


# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------

def make_train_env(args: Args, seed: int):
    def thunk():
        env = gym.make(f"dm_control/{args.task}-v0", render_mode="rgb_array")
        env = gym.wrappers.RecordEpisodeStatistics(env)
        from envs import PixelObservationWrapper, FrameStackWrapper, NormalizePixels
        env = PixelObservationWrapper(env, image_size=args.image_size)
        env = FrameStackWrapper(env, k=args.frame_stack)
        env = NormalizePixels(env)
        env.action_space.seed(seed)
        return env
    return thunk


@torch.no_grad()
def evaluate_ood(
    actor: Actor,
    encoder: PixelEncoder,
    qlae: QLAE,
    device: torch.device,
    args: Args,
    n_episodes: int = 10,
) -> tuple[float, float]:
    """Run n episodes on DistractingCS color-hard, return (mean, std) return."""
    dcs_task = args.task.replace("-", "_")
    env = make_dcs(dcs_task, distraction="color", difficulty="hard", seed=42,
                   image_size=args.image_size, frame_stack=args.frame_stack)
    returns = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        ep_ret, done = 0.0, False
        while not done:
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(device)
            z, _  = encode_frames(obs_t, encoder, qlae, detach_encoder=False)
            action = actor.get_mean_action(z).cpu().numpy()[0]
            obs, r, term, trunc, _ = env.step(action)
            ep_ret += r
            done = term or trunc
        returns.append(ep_ret)
    env.close()
    return float(np.mean(returns)), float(np.std(returns))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _maybe_set_wandb_key() -> None:
    if os.environ.get("WANDB_API_KEY"):
        return
    key_path = Path(__file__).parent / "WANDB_API_KEY.md"
    if key_path.exists():
        key = key_path.read_text().strip()
        if key:
            os.environ["WANDB_API_KEY"] = key


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args: Args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device(
        "cuda" if torch.cuda.is_available() and args.cuda
        else "mps"  if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Device: {device} | Task: {args.task} | Seed: {args.seed}")
    print(f"ALDA: n_z={args.n_z}  K={args.codebook_size}  β={args.beta}  "
          f"λ_enc={args.lambda_enc}  policy_dim={args.policy_dim}")

    # ---- Logging ----
    _maybe_set_wandb_key()
    if args.track:
        import wandb
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.run_name,
            config=vars(args),
            sync_tensorboard=True,
            save_code=True,
            tags=["alda", args.task, f"seed{args.seed}"],
            group=f"alda__{args.task}",
        )
    writer = SummaryWriter(f"runs/{args.run_name}")
    ckpt_dir = Path(args.checkpoint_dir) / args.run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ---- Environment ----
    envs     = gym.vector.SyncVectorEnv([make_train_env(args, args.seed)])
    act_dim  = int(np.prod(envs.single_action_space.shape))
    obs_shape = envs.single_observation_space.shape   # (9, 84, 84)

    action_scale = torch.tensor(
        (envs.single_action_space.high - envs.single_action_space.low) / 2.0,
        dtype=torch.float32,
    )
    action_bias = torch.tensor(
        (envs.single_action_space.high + envs.single_action_space.low) / 2.0,
        dtype=torch.float32,
    )

    # ---- Models ----
    # Encoder/decoder operate on single RGB frames (3 channels), not stacked (9).
    encoder = PixelEncoder(in_channels=C_PER_FRAME, latent_dim=args.n_z).to(device)
    decoder = PixelDecoder(out_channels=C_PER_FRAME, latent_dim=args.n_z).to(device)
    qlae    = QLAE(n_z=args.n_z, K=args.codebook_size, beta=args.beta).to(device)

    actor = Actor(args.policy_dim, act_dim, action_scale, action_bias).to(device)
    qf1   = SoftQNetwork(args.policy_dim, act_dim).to(device)
    qf2   = SoftQNetwork(args.policy_dim, act_dim).to(device)
    qf1_t = SoftQNetwork(args.policy_dim, act_dim).to(device)
    qf2_t = SoftQNetwork(args.policy_dim, act_dim).to(device)
    qf1_t.load_state_dict(qf1.state_dict())
    qf2_t.load_state_dict(qf2.state_dict())

    # ---- Optimisers ----
    # AdamW on encoder/decoder: the weight_decay parameter IS the λ_θ, λ_φ from Eq. 7.
    # Decision D3: AdamW applies WD at each .step() call; since enc_opt.step() is
    # called twice per global step (steps 1 and 2), effective decay = 2 × lambda_enc.
    # This is a minor over-regularisation accepted for simplicity.
    enc_opt   = optim.AdamW(encoder.parameters(), lr=args.recon_lr,
                            weight_decay=args.lambda_enc)
    dec_opt   = optim.AdamW(decoder.parameters(), lr=args.recon_lr,
                            weight_decay=args.lambda_dec)
    # Codebook uses plain Adam — no weight decay (prototypes should not be penalised to 0).
    cb_opt    = optim.Adam(qlae.parameters(), lr=args.recon_lr)
    qf_opt    = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr)
    actor_opt = optim.Adam(actor.parameters(), lr=args.policy_lr)

    if args.autotune:
        target_entropy = float(-act_dim)
        log_alpha  = torch.zeros(1, requires_grad=True, device=device)
        alpha      = log_alpha.exp().item()
        alpha_opt  = optim.Adam([log_alpha], lr=args.q_lr)
    else:
        alpha = args.alpha

    # ---- Replay buffer ----
    # Stored as (9, 84, 84) uint8 — same format as SAC+AE.  encode_frames() splits
    # the 9-channel tensor into k=3 single-frame (3, 84, 84) inputs at training time.
    rb = PixelReplayBuffer(obs_shape, act_dim, args.buffer_size, device)

    # -----------------------------------------------------------------------
    # Main loop
    # -----------------------------------------------------------------------
    obs, _ = envs.reset(seed=args.seed)
    start  = time.time()

    # Sentinel values for logging (updated each training step)
    actor_loss  = torch.tensor(0.0)
    assoc_loss  = torch.tensor(0.0)
    recon_loss  = torch.tensor(0.0)
    qf_loss     = torch.tensor(0.0)

    for step in range(args.total_timesteps):

        # ---- Collect ----
        if step < args.learning_starts:
            actions = np.array([envs.single_action_space.sample()])
        else:
            encoder.eval()
            qlae.eval()
            with torch.no_grad():
                obs_t  = torch.FloatTensor(obs).to(device)
                z, _   = encode_frames(obs_t, encoder, qlae)
                actions_t, _, _ = actor.get_action(z)
            actions = actions_t.cpu().numpy()
            encoder.train()
            qlae.train()

        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # Episode stats (gymnasium 1.x)
        if "episode" in infos:
            mask = infos.get("_episode", np.ones(1, dtype=bool))
            for idx in np.where(mask)[0]:
                ep_r = float(infos["episode"]["r"][idx])
                ep_l = int(infos["episode"]["l"][idx])
                print(f"step={step:>7d}  train_return={ep_r:>8.1f}  len={ep_l}")
                writer.add_scalar("charts/train_return",    ep_r, step)
                writer.add_scalar("charts/episode_length",  ep_l, step)

        # Replay buffer — termination (not truncation) as done flag
        real_next = next_obs.copy()
        if "final_observation" in infos:
            for idx, trunc in enumerate(truncations):
                if trunc:
                    real_next[idx] = infos["final_observation"][idx]

        rb.add(
            obs[0], real_next[0],
            actions[0].astype(np.float32),
            float(rewards[0]),
            float(terminations[0]),
        )
        obs = next_obs

        if step < args.learning_starts or len(rb) < args.batch_size:
            continue

        # ---- Train ----
        obs_b, next_b, act_b, rew_b, done_b = rb.sample(args.batch_size)

        # ── Step 1: Critic + encoder update ──────────────────────────────────
        # Gradient flows from critic loss into encoder. Codebook accumulates
        # gradient here but cb_opt is NOT stepped (cleared at start of step 2).
        # Decision D1: codebook not updated by critic loss.
        with torch.no_grad():
            z_next, _ = encode_frames(next_b, encoder, qlae)
            na, log_pi, _ = actor.get_action(z_next)
            q_next = torch.min(qf1_t(z_next, na), qf2_t(z_next, na)) - alpha * log_pi
            q_tgt  = rew_b.unsqueeze(1) + (1 - done_b.unsqueeze(1)) * args.gamma * q_next

        z_b, _ = encode_frames(obs_b, encoder, qlae)
        qf1_loss = F.mse_loss(qf1(z_b, act_b), q_tgt)
        qf2_loss = F.mse_loss(qf2(z_b, act_b), q_tgt)
        qf_loss  = qf1_loss + qf2_loss

        enc_opt.zero_grad()
        qf_opt.zero_grad()
        qf_loss.backward()
        qf_opt.step()
        enc_opt.step()
        # Note: codebook has accumulated gradient from qf_loss, but is NOT stepped.

        # ── Step 2: Auxiliary losses (encoder, decoder, codebook) ─────────────
        # L_assoc: encoder → codebook (StopGrad on codebook, per Eq. 7)
        # L_recon: decoder + codebook (z_s not detached → codebook gets gradient)
        # Decision D1: cb_opt.zero_grad() clears the step-1 codebook gradient.
        enc_opt.zero_grad()
        dec_opt.zero_grad()
        cb_opt.zero_grad()

        B = obs_b.shape[0]
        k = args.frame_stack
        frames_b = obs_b.view(B * k, C_PER_FRAME,
                              obs_b.shape[2], obs_b.shape[3])   # (B*k, 3, H, W)

        z_enc_aux = encoder(frames_b)             # (B*k, n_z); gradient ON for encoder
        z_s_aux   = qlae(z_enc_aux)               # (B*k, n_z); gradient ON for codebook

        # Association loss: push encoder toward codebook (codebook is stop-gradiented)
        assoc_loss = qlae.association_loss(z_enc_aux)   # gradient into encoder only

        # Reconstruction loss: decoder reconstructs single frames from z_s
        # z_s_aux is NOT detached → gradient flows into decoder AND codebook AND encoder
        recon       = decoder(z_s_aux)                  # (B*k, 3, H, W)
        recon_loss  = F.mse_loss(recon, frames_b)

        aux_loss = assoc_loss + recon_loss
        aux_loss.backward()
        enc_opt.step()   # encoder: from L_assoc + L_recon + AdamW WD
        dec_opt.step()   # decoder: from L_recon + AdamW WD
        cb_opt.step()    # codebook: from L_recon only (L_assoc has sg on codebook)

        # ── Step 3: Actor + alpha (encoder and QLAE fully detached) ───────────
        if step % args.policy_frequency == 0:
            for _ in range(args.policy_frequency):
                z_det, _ = encode_frames(obs_b, encoder, qlae, detach_encoder=True)
                pi, log_pi, _ = actor.get_action(z_det)
                actor_loss = (alpha * log_pi
                              - torch.min(qf1(z_det, pi), qf2(z_det, pi))).mean()

                actor_opt.zero_grad()
                actor_loss.backward()
                actor_opt.step()

                if args.autotune:
                    with torch.no_grad():
                        _, log_pi, _ = actor.get_action(z_det)
                    alpha_loss = (-log_alpha.exp() * (log_pi + target_entropy)).mean()
                    alpha_opt.zero_grad()
                    alpha_loss.backward()
                    alpha_opt.step()
                    alpha = log_alpha.exp().item()

        # ── Step 4: Target network soft update ────────────────────────────────
        if step % args.target_frequency == 0:
            for p, tp in zip(qf1.parameters(), qf1_t.parameters()):
                tp.data.copy_(args.tau * p.data + (1 - args.tau) * tp.data)
            for p, tp in zip(qf2.parameters(), qf2_t.parameters()):
                tp.data.copy_(args.tau * p.data + (1 - args.tau) * tp.data)

        # ---- Logging ----
        if step % 1_000 == 0:
            sps = int(step / (time.time() - start)) if step > 0 else 0
            cb_usage = qlae.codebook_usage(z_enc_aux.detach())
            writer.add_scalar("losses/qf_loss",     qf_loss.item(),    step)
            writer.add_scalar("losses/recon_loss",   recon_loss.item(), step)
            writer.add_scalar("losses/assoc_loss",   assoc_loss.item(), step)
            writer.add_scalar("losses/actor_loss",   actor_loss.item(), step)
            writer.add_scalar("losses/alpha",        alpha,             step)
            writer.add_scalar("losses/cb_usage",     cb_usage,          step)
            writer.add_scalar("charts/SPS",          sps,               step)

        # ---- OOD evaluation + checkpoint ----
        if step > 0 and step % args.eval_frequency == 0:
            encoder.eval(); qlae.eval(); actor.eval()
            mean_r, std_r = evaluate_ood(
                actor, encoder, qlae, device, args, args.n_eval_episodes
            )
            encoder.train(); qlae.train(); actor.train()
            print(f"step={step:>7d}  OOD color-hard  return={mean_r:>7.1f} ± {std_r:.1f}")
            writer.add_scalar("eval/ood_color_hard_return", mean_r, step)
            writer.add_scalar("eval/ood_color_hard_std",    std_r,  step)

            ckpt_path = ckpt_dir / f"step_{step:07d}.pt"
            torch.save({
                "step":         step,
                "args":         vars(args),
                "encoder":      encoder.state_dict(),
                "qlae":         qlae.state_dict(),
                "decoder":      decoder.state_dict(),
                "actor":        actor.state_dict(),
                "action_scale": actor.action_scale.cpu(),
                "action_bias":  actor.action_bias.cpu(),
            }, ckpt_path)
            print(f"  saved → {ckpt_path}")

    envs.close()
    writer.close()
    if args.track:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    args = tyro.cli(Args)
    train(args)
