# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository implements **ALDA (Associative Latent DisentAnglement)** — an algorithm for zero-shot out-of-distribution generalization in vision-based Reinforcement Learning, without data augmentation. The implementation is a MIPT internship test assignment.

The approach:
- Learns a **disentangled latent representation** via QLAE (Quantized Latent AutoEncoder)
- Equips it with an **associative memory mechanism** to remap OOD visual factors back to in-distribution values
- Built on top of **Soft Actor-Critic (SAC)**

Training environment: **DeepMind Control Suite (DMC)**  
Evaluation environment: **Distracting Control Suite** (OOD benchmark)

Reference implementation to start from: CleanRL's SAC — `https://github.com/vwxyzjn/cleanrl`  
Core paper: *Zero-Shot Generalization of Vision-Based RL Without Data Augmentation* (Batra & Sukhatme, ICML 2024)

## Setup

```bash
uv sync
```

Installed dependencies: `torch`, `gymnasium`, `tyro`, `tensorboard`, `numpy`.  
Vendored: `cleanrl_utils/` (ReplayBuffer from CleanRL — not on PyPI).

## Running SAC

```bash
# Verified working: Pendulum-v1 (classic control, no MuJoCo needed)
uv run python sac_continuous_action.py --env-id Pendulum-v1 --total-timesteps 50000 --no-cuda

# Logs written to runs/<run_name>/; view with:
uv run tensorboard --logdir runs
```

## Gymnasium 1.x compatibility

`sac_continuous_action.py` has been patched for gymnasium >= 1.0. Two breaking changes from the original CleanRL code:
- Episode stats: `infos["final_info"]` gone — use `infos["episode"]["r"]` with `infos["_episode"]` mask
- Terminal obs: `infos["final_observation"]` not set for truncated episodes — guarded with `if "final_observation" in infos`

## Architecture Plan

The codebase should be structured around these components:

### Core Algorithm (`alda/`)
- **`encoder.py`** — CNN encoder mapping pixel observations to latent space
- **`qlae.py`** — Quantized Latent AutoEncoder for disentanglement (implements QLAE from Hsu et al. 2023); produces factored discrete/continuous codes
- **`memory.py`** — Associative memory module; stores in-distribution latent codes and retrieves nearest neighbors to remap OOD observations
- **`sac.py`** — SAC agent (actor + twin critics + entropy tuning); operates on latent representations from QLAE
- **`alda.py`** — Top-level agent combining encoder + QLAE + associative memory + SAC

### Infrastructure
- **`train.py`** — Training loop on DMC tasks
- **`eval.py`** — Evaluation loop on Distracting Control Suite
- **`replay_buffer.py`** — Off-policy replay buffer storing pixel observations
- **`envs.py`** — Environment wrappers for DMC and Distracting Control Suite

### Key Algorithmic Details
- QLAE disentangles latent space into **task-relevant** and **task-irrelevant** factors
- During training, associative memory is populated with in-distribution latent codes
- At test time (OOD), task-irrelevant factors are remapped via nearest-neighbor lookup in associative memory before passing to the policy
- SAC is the underlying RL algorithm; actor and critic are conditioned on the task-relevant latent factor only
