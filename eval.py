"""
Standalone evaluation for SAC+AE and ALDA checkpoints.

Auto-detects algorithm from checkpoint keys:
  - "qlae" key present → ALDA  (encoder sees single frames, qlae remaps before policy)
  - no "qlae" key      → SAC+AE (encoder sees stacked 9-channel obs)

Evaluates on:
  - Clean DMC (in-distribution)
  - DistractingCS color / camera (easy / medium / hard)  — no extra data needed
  - DistractingCS background — needs DAVIS 2017 (~6 GB), skipped otherwise

Run:
    MUJOCO_GL=glfw uv run python eval.py \\
        --checkpoint checkpoints/<run>/step_0500000.pt \\
        --n-eval-episodes 10 --track
"""

import os
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from alda.encoder import PixelEncoder
from alda.qlae import QLAE
from envs import make_dmc, make_dcs


# ---------------------------------------------------------------------------
# Actor (shared by both algos; only input dim differs: 50 for SAC+AE, 36 for ALDA)
# ---------------------------------------------------------------------------

LOG_STD_MAX, LOG_STD_MIN = 2, -5
C_PER_FRAME = 3


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

    @torch.no_grad()
    def get_mean_action(self, z: torch.Tensor) -> torch.Tensor:
        mean, _ = self(z)
        return torch.tanh(mean) * self.action_scale + self.action_bias


# ---------------------------------------------------------------------------
# Encoding: SAC+AE vs ALDA
# ---------------------------------------------------------------------------

def encode_sac_ae(
    obs: torch.Tensor,          # (B, k*C, H, W) already stacked
    encoder: PixelEncoder,
    qlae: None,
) -> torch.Tensor:
    """SAC+AE: encode stacked obs directly → (B, latent_dim)."""
    return encoder(obs)


def encode_alda(
    obs: torch.Tensor,          # (B, k*C, H, W)
    encoder: PixelEncoder,
    qlae: QLAE,
) -> torch.Tensor:
    """ALDA: encode each frame separately, soft-retrieve, concat → (B, k*n_z)."""
    B = obs.shape[0]
    k = obs.shape[1] // C_PER_FRAME
    frames = obs.view(B * k, C_PER_FRAME, obs.shape[2], obs.shape[3])
    z_enc  = encoder(frames)
    z_s    = qlae(z_enc)
    return z_s.view(B, k * qlae.n_z)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_episodes(
    actor: Actor,
    encoder: PixelEncoder,
    qlae,                       # QLAE | None
    encode_fn,                  # encode_sac_ae | encode_alda
    env,
    device: torch.device,
    n_episodes: int,
) -> tuple[float, float]:
    returns = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        ep_ret, done = 0.0, False
        while not done:
            obs_t  = torch.FloatTensor(obs).unsqueeze(0).to(device)
            z      = encode_fn(obs_t, encoder, qlae)
            action = actor.get_mean_action(z).cpu().numpy()[0]
            obs, r, term, trunc, _ = env.step(action)
            ep_ret += r
            done = term or trunc
        returns.append(ep_ret)
    return float(np.mean(returns)), float(np.std(returns))


def evaluate_all(
    actor, encoder, qlae, encode_fn, device,
    task, image_size, frame_stack, n_episodes,
    distractions, difficulties, background_dataset_path,
) -> list[dict]:
    results = []
    dcs_task = task.replace("-", "_")

    # In-distribution
    env = make_dmc(task, seed=42, image_size=image_size, frame_stack=frame_stack)
    mean_r, std_r = run_episodes(actor, encoder, qlae, encode_fn, env, device, n_episodes)
    env.close()
    results.append({"setting": "clean", "distraction": "none", "difficulty": "none",
                    "mean_return": mean_r, "std_return": std_r})
    print(f"  clean                  {mean_r:>8.1f} ± {std_r:.1f}")

    # OOD
    for dist in distractions:
        if dist == "background" and background_dataset_path is None:
            print("  [skip] background — pass --background-dataset-path")
            continue
        for diff in difficulties:
            try:
                env = make_dcs(
                    dcs_task, distraction=dist, difficulty=diff, seed=42,
                    image_size=image_size, frame_stack=frame_stack,
                    background_dataset_path=background_dataset_path,
                )
                mean_r, std_r = run_episodes(
                    actor, encoder, qlae, encode_fn, env, device, n_episodes
                )
                env.close()
            except Exception as e:
                print(f"  [error] {dist}-{diff}: {e}")
                mean_r, std_r = float("nan"), float("nan")
            results.append({"setting": f"{dist}-{diff}", "distraction": dist,
                            "difficulty": diff, "mean_return": mean_r, "std_return": std_r})
            print(f"  {dist:<10} {diff:<8}  {mean_r:>8.1f} ± {std_r:.1f}")

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _maybe_set_wandb_key() -> None:
    if os.environ.get("WANDB_API_KEY"):
        return
    key_path = Path(__file__).parent / "WANDB_API_KEY.md"
    if key_path.exists():
        key = key_path.read_text().strip()
        if key:
            os.environ["WANDB_API_KEY"] = key


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--n-eval-episodes", type=int, default=10)
    parser.add_argument("--distractions", nargs="+", default=["color", "camera"])
    parser.add_argument("--difficulties", nargs="+", default=["easy", "medium", "hard"])
    parser.add_argument("--background-dataset-path", default=None)
    parser.add_argument("--track", action="store_true")
    parser.add_argument("--wandb-project", default="alda-eval")
    parser.add_argument("--wandb-entity", default=None)
    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Device: {device}")

    # ---- Load checkpoint ----
    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg  = ckpt["args"]
    task        = cfg["task"]
    image_size  = cfg["image_size"]
    frame_stack = cfg["frame_stack"]
    step        = ckpt["step"]
    is_alda     = "qlae" in ckpt

    algo_tag = "alda" if is_alda else "sac_ae"
    print(f"Checkpoint: {args.checkpoint}")
    print(f"  algo={algo_tag}  task={task}  step={step:,}")

    action_scale = ckpt["action_scale"].to(device)
    action_bias  = ckpt["action_bias"].to(device)
    act_dim      = action_scale.shape[0]

    if is_alda:
        # ALDA: encoder sees single frames (in_channels=3), policy_dim = k*n_z
        n_z        = cfg["n_z"]
        K          = cfg["codebook_size"]
        beta       = cfg["beta"]
        policy_dim = frame_stack * n_z

        encoder = PixelEncoder(in_channels=C_PER_FRAME, latent_dim=n_z).to(device)
        qlae    = QLAE(n_z=n_z, K=K, beta=beta).to(device)
        actor   = Actor(policy_dim, act_dim, action_scale, action_bias).to(device)
        encoder.load_state_dict(ckpt["encoder"])
        qlae.load_state_dict(ckpt["qlae"])
        actor.load_state_dict(ckpt["actor"])
        encode_fn = encode_alda
        print(f"  n_z={n_z}  K={K}  β={beta}  policy_dim={policy_dim}")
    else:
        # SAC+AE: encoder sees stacked obs (in_channels=k*3), policy_dim = latent_dim
        latent_dim = cfg["latent_dim"]
        in_channels = frame_stack * C_PER_FRAME

        encoder = PixelEncoder(in_channels=in_channels, latent_dim=latent_dim).to(device)
        qlae    = None
        actor   = Actor(latent_dim, act_dim, action_scale, action_bias).to(device)
        encoder.load_state_dict(ckpt["encoder"])
        actor.load_state_dict(ckpt["actor"])
        encode_fn = encode_sac_ae
        print(f"  latent_dim={latent_dim}")

    encoder.eval()
    actor.eval()
    if qlae is not None:
        qlae.eval()

    # ---- Wandb ----
    _maybe_set_wandb_key()
    run = None
    if args.track:
        import wandb
        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=f"eval__{algo_tag}__{task}__step{step}",
            config={
                "checkpoint": args.checkpoint, "algo": algo_tag,
                "task": task, "step": step,
                "n_eval_episodes": args.n_eval_episodes,
            },
            tags=[algo_tag, task],
        )

    # ---- Evaluate ----
    print(f"\nEvaluating {task} ({algo_tag}) — {args.n_eval_episodes} episodes each")
    print(f"  {'setting':<22} {'mean':>8}   std")
    print("  " + "-" * 38)

    results = evaluate_all(
        actor=actor, encoder=encoder, qlae=qlae, encode_fn=encode_fn,
        device=device, task=task, image_size=image_size, frame_stack=frame_stack,
        n_episodes=args.n_eval_episodes, distractions=args.distractions,
        difficulties=args.difficulties,
        background_dataset_path=args.background_dataset_path,
    )

    # ---- Log ----
    if run is not None:
        import wandb
        table = wandb.Table(
            columns=["algo", "setting", "distraction", "difficulty",
                     "mean_return", "std_return"]
        )
        for r in results:
            table.add_data(algo_tag, r["setting"], r["distraction"],
                           r["difficulty"], r["mean_return"], r["std_return"])
        run.log({"eval/results_table": table, "step": step})
        for r in results:
            key = r["setting"].replace("-", "_")
            run.log({f"eval/{key}_return": r["mean_return"],
                     f"eval/{key}_std":    r["std_return"]})
        run.finish()

    print("\nDone.")


if __name__ == "__main__":
    main()
