from __future__ import annotations
import numpy as np
import torch

from env import HumanoidPaperEnv
from policy import GaussianPolicy
from trajectory import Trajectory
from tqdm import tqdm


def rollout_policy(
    env: HumanoidPaperEnv,
    policy: GaussianPolicy,
    init_state: np.ndarray,
    horizon: int,
    device: torch.device,
    deterministic: bool = False,
    source_name: str = "policy",
    source_id: str = "policy",
) -> Trajectory:
    state_dim = init_state.shape[0]
    act_dim = env.act_dim

    sim_states = np.zeros((horizon + 1, state_dim), dtype=np.float64)
    obs = np.zeros((horizon, env.obs_dim), dtype=np.float32)
    actions = np.zeros((horizon, act_dim), dtype=np.float64)
    rewards = np.zeros(horizon, dtype=np.float64)
    dones = np.zeros(horizon, dtype=bool)

    sim_states[0] = init_state.copy()

    for t in range(horizon):
        o = env.policy_obs_from_sim_state(sim_states[t])
        obs[t] = o

        if deterministic:
            a = policy.mean_action(o)
        else:
            a = policy.sample_action(o)

        a = np.clip(a, env.env.action_space.low, env.env.action_space.high)
        next_state, r, d, _ = env.step_from_state(sim_states[t], a)

        sim_states[t + 1] = next_state
        actions[t] = a
        rewards[t] = r
        dones[t] = d

        if d:
            for tt in range(t + 1, horizon):
                sim_states[tt + 1] = next_state
                obs[tt] = env.policy_obs_from_sim_state(next_state)
                actions[tt] = 0.0
                rewards[tt] = 0.0
                dones[tt] = True
            break

    return Trajectory(
    sim_states=sim_states,
    obs=obs,
    actions=actions,
    rewards=rewards,
    dones=dones,
    source_name=source_name,
    source_type="policy",
    source_id=source_id,
)


def evaluate_policy(
    env: HumanoidPaperEnv,
    policy: GaussianPolicy,
    init_state: np.ndarray,
    horizon: int,
    device: torch.device,
    n_rollouts: int = 5,
    deterministic: bool = True,
):
    returns = []
    trajs = []
    for i in range(n_rollouts):
        traj = rollout_policy(
        env=env,
        policy=policy,
        init_state=init_state,
        horizon=horizon,
        device=device,
        deterministic=deterministic,
        source_name=f"eval_{i}",
        source_id=f"eval_{i}",
    )
        trajs.append(traj)
        returns.append(float(np.sum(traj.rewards)))
    return {
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "returns": returns,
        "trajectories": trajs,
    }