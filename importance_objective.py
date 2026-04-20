from __future__ import annotations
from typing import List, Dict
import numpy as np
import torch

from policy import GaussianPolicy
from trajectory import Trajectory, LinearGaussianController
from guide import fused_mixture_logprob_prefixes


def build_sample_cache(
    trajectories: List[Trajectory],
    controllers: List[LinearGaussianController],
    policy_bank: Dict[str, GaussianPolicy],
) -> List[Dict]:
    """
    Precompute q-prefixes for each trajectory.
    Policy prefixes are recomputed during optimization because theta changes.
    """
    cache = []

    for traj in trajectories:
        q_prefix_log = fused_sampling_logprob_prefixes(controllers, policy_bank, traj)
        cache.append(
            {
                "traj": traj,
                "q_prefix_log": q_prefix_log,
            }
        )
    return cache

def fused_sampling_logprob_prefixes(
    controllers: List[LinearGaussianController],
    policy_bank: Dict[str, GaussianPolicy],
    traj: Trajectory,
) -> np.ndarray:
    parts = []

    if len(controllers) > 0:
        parts.append(fused_mixture_logprob_prefixes(controllers, traj))

    for pid, pol in policy_bank.items():
        logp = pol.log_prob_prefixes_for_traj(traj.obs, traj.actions)
        parts.append(logp)

    if len(parts) == 0:
        raise ValueError("No proposal distributions available.")

    all_logs = np.stack(parts, axis=0)   # [M, T]
    out = np.zeros(all_logs.shape[1], dtype=np.float64)

    for t in range(all_logs.shape[1]):
        vals = all_logs[:, t]
        m = np.max(vals)
        out[t] = m + np.log(np.mean(np.exp(vals - m)))

    return out

def phi_objective(
    policy: GaussianPolicy,
    cache: List[Dict],
    wr: float,
    device: torch.device,
) -> torch.Tensor:
    T = cache[0]["traj"].actions.shape[0]
    phi = torch.zeros((), dtype=torch.float32, device=device)

    traj_logpi_prefixes = []
    traj_rewards = []
    traj_q_prefixes = []

    for item in cache:
        traj = item["traj"]
        obs_t = torch.as_tensor(traj.obs, dtype=torch.float32, device=device)
        act_t = torch.as_tensor(traj.actions, dtype=torch.float32, device=device)
        step_logp = policy.log_prob_actions(obs_t, act_t)
        prefix_logp = torch.cumsum(step_logp, dim=0)  # [T]

        traj_logpi_prefixes.append(prefix_logp)
        traj_rewards.append(torch.as_tensor(traj.rewards, dtype=torch.float32, device=device))
        traj_q_prefixes.append(torch.as_tensor(item["q_prefix_log"], dtype=torch.float32, device=device))

    for t in range(T):
        logw_t = torch.stack(
            [traj_logpi_prefixes[i][t] - traj_q_prefixes[i][t] for i in range(len(cache))],
            dim=0,
        )  # [N]
        r_t = torch.stack([traj_rewards[i][t] for i in range(len(cache))], dim=0)

        logZt = torch.logsumexp(logw_t, dim=0)
        w_norm = torch.exp(logw_t - logZt)
        phi = phi + torch.sum(w_norm * r_t) + wr * logZt
    return phi

def estimate_return_objective(
    policy: GaussianPolicy,
    cache: List[Dict],
    device: torch.device,
) -> float:
    T = cache[0]["traj"].actions.shape[0]
    estimate = torch.zeros((), dtype=torch.float32, device=device)

    traj_logpi_prefixes = []
    traj_rewards = []
    traj_q_prefixes = []

    for item in cache:
        traj = item["traj"]
        obs_t = torch.as_tensor(traj.obs, dtype=torch.float32, device=device)
        act_t = torch.as_tensor(traj.actions, dtype=torch.float32, device=device)
        step_logp = policy.log_prob_actions(obs_t, act_t)
        prefix_logp = torch.cumsum(step_logp, dim=0)

        traj_logpi_prefixes.append(prefix_logp)
        traj_rewards.append(torch.as_tensor(traj.rewards, dtype=torch.float32, device=device))
        traj_q_prefixes.append(torch.as_tensor(item["q_prefix_log"], dtype=torch.float32, device=device))

    for t in range(T):
        logw_t = torch.stack(
            [traj_logpi_prefixes[i][t] - traj_q_prefixes[i][t] for i in range(len(cache))],
            dim=0,
        )
        r_t = torch.stack([traj_rewards[i][t] for i in range(len(cache))], dim=0)

        logZt = torch.logsumexp(logw_t, dim=0)
        w_norm = torch.exp(logw_t - logZt)

        w_norm = torch.clamp(w_norm, min=1e-4, max=10.0)
        w_norm = w_norm / (torch.sum(w_norm) + 1e-8)

        estimate = estimate + torch.sum(w_norm * r_t)

    return float(estimate.item())

def select_active_set(
    trajectories: List[Trajectory],
    controllers: List[LinearGaussianController],
    policy_bank: Dict[str, GaussianPolicy],
    current_best_policy: GaussianPolicy,
    active_set_size: int,
    device: torch.device,
) -> List[Trajectory]:
    """
    Keep:
    - all guide trajectories first
    - then top policy / mixed samples by approximate importance score
    """
    if len(trajectories) <= active_set_size:
        return trajectories

    scored = []
    for traj in trajectories:
        q_prefix_log = fused_sampling_logprob_prefixes(controllers, policy_bank, traj)
        logpi_prefix = current_best_policy.log_prob_prefixes_for_traj(traj.obs, traj.actions)
        score = float(np.mean(logpi_prefix - q_prefix_log))
        is_guide = traj.source_type == "guide"
        scored.append((is_guide, score, traj))

    guides = [x[2] for x in scored if x[0]]
    non_guides = [x for x in scored if not x[0]]
    non_guides.sort(key=lambda z: z[1], reverse=True)

    remaining = max(0, active_set_size - len(guides))
    selected = guides + [x[2] for x in non_guides[:remaining]]
    return selected