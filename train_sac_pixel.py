"""
SAC + autoencoder (SAC+AE) on DMC pixel observations.

Trains on clean DMC pixels; evaluates periodically on DistractingCS color-hard
to show the generalization gap that ALDA will later close.

Run:
    MUJOCO_GL=glfw uv run python train_sac_pixel.py --task walker-walk --seed 1 --track
    MUJOCO_GL=glfw uv run python train_sac_pixel.py --task cartpole-balance --seed 1 --track

Requires WANDB_API_KEY when --track is set.
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
    """Directory to save encoder+actor checkpoints."""

    # Encoder / decoder
    latent_dim: int = 50
    frame_stack: int = 3
    image_size: int = 84
    recon_lr: float = 1e-3

    # SAC
    buffer_size: int = 100_000
    """Replay buffer capacity. Each slot ≈ 2×9×84×84 bytes (uint8). 100k ≈ 12 GB RAM."""
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
        self.run_name = f"sac_pixel__{self.task}__seed{self.seed}__{int(time.time())}"


# ---------------------------------------------------------------------------
# Networks
# ---------------------------------------------------------------------------

LOG_STD_MAX, LOG_STD_MIN = 2, -5


class Actor(nn.Module):
    def __init__(self, latent_dim: int, act_dim: int,
                 action_scale: torch.Tensor, action_bias: torch.Tensor):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.ReLU(),
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
        """Deterministic action for evaluation."""
        mean, _ = self(z)
        return torch.tanh(mean) * self.action_scale + self.action_bias


class SoftQNetwork(nn.Module):
    def __init__(self, latent_dim: int, act_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + act_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([z, a], dim=1))


# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------

def make_train_env(args: Args, seed: int):
    """Single DMC env with pixel obs, suitable for SyncVectorEnv."""
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
    device: torch.device,
    args: Args,
    n_episodes: int = 10,
) -> tuple[float, float]:
    """Run episodes on DistractingCS color-hard, return (mean, std) return."""
    dcs_task = args.task.replace("-", "_")
    env = make_dcs(dcs_task, distraction="color", difficulty="hard", seed=42,
                   image_size=args.image_size, frame_stack=args.frame_stack)
    returns = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        ep_ret, done = 0.0, False
        while not done:
            z = encoder(torch.FloatTensor(obs).unsqueeze(0).to(device))
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
    """Read WANDB_API_KEY from file if not already set in environment."""
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

    # Logging
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
            tags=["sac_ae", args.task, f"seed{args.seed}"],
            group=f"sac_ae__{args.task}",
        )
    writer = SummaryWriter(f"runs/{args.run_name}")
    ckpt_dir = Path(args.checkpoint_dir) / args.run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Environment
    envs = gym.vector.SyncVectorEnv([make_train_env(args, args.seed)])
    act_dim   = int(np.prod(envs.single_action_space.shape))
    obs_shape = envs.single_observation_space.shape  # (9, 84, 84)

    action_scale = torch.tensor(
        (envs.single_action_space.high - envs.single_action_space.low) / 2.0,
        dtype=torch.float32,
    )
    action_bias = torch.tensor(
        (envs.single_action_space.high + envs.single_action_space.low) / 2.0,
        dtype=torch.float32,
    )

    # Models
    in_channels = obs_shape[0]  # C*k = 9
    encoder = PixelEncoder(in_channels=in_channels, latent_dim=args.latent_dim).to(device)
    decoder = PixelDecoder(out_channels=in_channels, latent_dim=args.latent_dim).to(device)
    actor   = Actor(args.latent_dim, act_dim, action_scale, action_bias).to(device)
    qf1     = SoftQNetwork(args.latent_dim, act_dim).to(device)
    qf2     = SoftQNetwork(args.latent_dim, act_dim).to(device)
    qf1_t   = SoftQNetwork(args.latent_dim, act_dim).to(device)
    qf2_t   = SoftQNetwork(args.latent_dim, act_dim).to(device)
    qf1_t.load_state_dict(qf1.state_dict())
    qf2_t.load_state_dict(qf2.state_dict())

    # Optimisers
    # Encoder is updated by: (a) critic loss, (b) reconstruction loss.
    # Actor uses encoder(detach=True) → no encoder gradient from actor.
    enc_opt   = optim.Adam(encoder.parameters(), lr=args.recon_lr)
    dec_opt   = optim.Adam(decoder.parameters(), lr=args.recon_lr)
    qf_opt    = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr)
    actor_opt = optim.Adam(actor.parameters(), lr=args.policy_lr)

    if args.autotune:
        target_entropy = float(-act_dim)
        log_alpha  = torch.zeros(1, requires_grad=True, device=device)
        alpha      = log_alpha.exp().item()
        alpha_opt  = optim.Adam([log_alpha], lr=args.q_lr)
    else:
        alpha = args.alpha

    # Replay buffer
    rb = PixelReplayBuffer(obs_shape, act_dim, args.buffer_size, device)

    # -----------------------------------------------------------------------
    # Main loop
    # -----------------------------------------------------------------------
    obs, _ = envs.reset(seed=args.seed)
    start  = time.time()

    for step in range(args.total_timesteps):

        # --- Collect ---
        if step < args.learning_starts:
            actions = np.array([envs.single_action_space.sample()])
        else:
            encoder.eval()
            with torch.no_grad():
                z = encoder(torch.FloatTensor(obs).to(device))
                actions_t, _, _ = actor.get_action(z)
            actions = actions_t.cpu().numpy()
            encoder.train()

        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # Episode stats (gymnasium 1.x)
        if "episode" in infos:
            mask = infos.get("_episode", np.ones(1, dtype=bool))
            for idx in np.where(mask)[0]:
                ep_r = float(infos["episode"]["r"][idx])
                ep_l = int(infos["episode"]["l"][idx])
                print(f"step={step:>7d}  train_return={ep_r:>8.1f}  len={ep_l}")
                writer.add_scalar("charts/train_return", ep_r, step)
                writer.add_scalar("charts/episode_length", ep_l, step)

        # Replay buffer — use termination (not truncation) for done flag
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

        # --- Train ---
        obs_b, next_b, act_b, rew_b, done_b = rb.sample(args.batch_size)

        # 1. Critic + encoder update (gradient flows from critic into encoder)
        with torch.no_grad():
            z_next = encoder(next_b)
            na, log_pi, _ = actor.get_action(z_next)
            q_next = torch.min(qf1_t(z_next, na), qf2_t(z_next, na)) - alpha * log_pi
            q_tgt  = rew_b.unsqueeze(1) + (1 - done_b.unsqueeze(1)) * args.gamma * q_next

        z = encoder(obs_b)
        qf1_loss = F.mse_loss(qf1(z, act_b), q_tgt)
        qf2_loss = F.mse_loss(qf2(z, act_b), q_tgt)
        qf_loss  = qf1_loss + qf2_loss

        enc_opt.zero_grad()
        qf_opt.zero_grad()
        qf_loss.backward()
        qf_opt.step()
        enc_opt.step()

        # 2. Reconstruction update (encoder + decoder)
        enc_opt.zero_grad()
        dec_opt.zero_grad()
        recon      = decoder(encoder(obs_b))
        recon_loss = F.mse_loss(recon, obs_b)
        recon_loss.backward()
        enc_opt.step()
        dec_opt.step()

        # 3. Actor + alpha update (encoder detached — no encoder grad from actor)
        if step % args.policy_frequency == 0:
            for _ in range(args.policy_frequency):
                z_det = encoder(obs_b, detach=True)
                pi, log_pi, _ = actor.get_action(z_det)
                actor_loss = (alpha * log_pi - torch.min(qf1(z_det, pi), qf2(z_det, pi))).mean()

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

        # 4. Target network soft update
        if step % args.target_frequency == 0:
            for p, tp in zip(qf1.parameters(), qf1_t.parameters()):
                tp.data.copy_(args.tau * p.data + (1 - args.tau) * tp.data)
            for p, tp in zip(qf2.parameters(), qf2_t.parameters()):
                tp.data.copy_(args.tau * p.data + (1 - args.tau) * tp.data)

        # --- Logging ---
        if step % 1_000 == 0:
            sps = int(step / (time.time() - start))
            writer.add_scalar("losses/qf_loss",    qf_loss.item(),    step)
            writer.add_scalar("losses/recon_loss",  recon_loss.item(), step)
            writer.add_scalar("losses/actor_loss",  actor_loss.item(), step)
            writer.add_scalar("losses/alpha",       alpha,             step)
            writer.add_scalar("charts/SPS",         sps,               step)

        # --- OOD evaluation + checkpoint ---
        if step > 0 and step % args.eval_frequency == 0:
            encoder.eval()
            actor.eval()
            mean_r, std_r = evaluate_ood(actor, encoder, device, args, args.n_eval_episodes)
            encoder.train()
            actor.train()
            print(f"step={step:>7d}  OOD color-hard  return={mean_r:>7.1f} ± {std_r:.1f}")
            writer.add_scalar("eval/ood_color_hard_return", mean_r, step)
            writer.add_scalar("eval/ood_color_hard_std",    std_r,  step)

            # Save checkpoint — enough to reconstruct policy for eval.py
            ckpt_path = ckpt_dir / f"step_{step:07d}.pt"
            torch.save({
                "step": step,
                "args": vars(args),
                "encoder": encoder.state_dict(),
                "actor":   actor.state_dict(),
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
