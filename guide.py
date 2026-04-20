from __future__ import annotations
import numpy as np
from typing import List

from env import HumanoidPaperEnv
from trajectory import LinearGaussianController, Trajectory
from tqdm import tqdm


def guide_mean_action(
    controller: LinearGaussianController,
    t: int,
    sim_state: np.ndarray,
) -> np.ndarray:
    dx = sim_state - controller.x_nom[t]
    return controller.u_nom[t] + controller.k[t] + controller.K[t] @ dx


def sample_guide_trajectory(
    env: HumanoidPaperEnv,
    controller: LinearGaussianController,
    init_state: np.ndarray,
    rng: np.random.Generator,
) -> Trajectory:
    T = controller.u_nom.shape[0]
    state_dim = init_state.shape[0]
    act_dim = controller.u_nom.shape[1]

    sim_states = np.zeros((T + 1, state_dim), dtype=np.float64)
    obs = np.zeros((T, env.obs_dim), dtype=np.float32)
    actions = np.zeros((T, act_dim), dtype=np.float64)
    rewards = np.zeros(T, dtype=np.float64)
    dones = np.zeros(T, dtype=bool)

    sim_states[0] = init_state.copy()

    for t in range(T):
        mean = guide_mean_action(controller, t, sim_states[t])
        cov = controller.cov[t]
        action = rng.multivariate_normal(mean=mean, cov=cov)
        action = np.clip(action, env.env.action_space.low, env.env.action_space.high)

        next_state, reward, done, ob = env.step_from_state(sim_states[t], action)

        sim_states[t + 1] = next_state
        obs[t] = ob
        actions[t] = action
        rewards[t] = reward
        dones[t] = done

        if done:
            for tt in range(t + 1, T):
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
        source_name=controller.source_name,
        source_type="guide",
        source_id=controller.source_name,
    )


def guide_logprob_prefixes(
    controller: LinearGaussianController,
    traj: Trajectory,
) -> np.ndarray:
    """
    log q(ζ_{1:t}) for t=1..T under the local linear-Gaussian guide.
    """
    T = traj.actions.shape[0]
    act_dim = traj.actions.shape[1]
    prefixes = np.zeros(T, dtype=np.float64)
    running = 0.0

    const = act_dim * np.log(2.0 * np.pi)

    for t in range(T):
        mean = guide_mean_action(controller, t, traj.sim_states[t])
        diff = traj.actions[t] - mean
        maha = diff.T @ controller.cov_inv[t] @ diff
        logp = -0.5 * (maha + controller.cov_logdet[t] + const)
        running += float(logp)
        prefixes[t] = running

    return prefixes


def fused_mixture_logprob_prefixes(
    controllers: List[LinearGaussianController],
    traj: Trajectory,
) -> np.ndarray:
    """
    q(ζ) = (1/n) sum_j q_j(ζ)
    For prefixes, do the same at each t:
    q(ζ_{1:t}) = (1/n) sum_j q_j(ζ_{1:t})
    """
    T = traj.actions.shape[0]
    n = len(controllers)

    all_prefix_logs = np.stack(
        [guide_logprob_prefixes(ctrl, traj) for ctrl in controllers],
        axis=0,
    )  # [n, T]

    out = np.zeros(T, dtype=np.float64)
    for t in range(T):
        vals = all_prefix_logs[:, t]
        m = np.max(vals)
        out[t] = m + np.log(np.mean(np.exp(vals - m)))
    return out