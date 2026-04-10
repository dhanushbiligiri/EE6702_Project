from __future__ import annotations
import numpy as np
import gymnasium as gym
from typing import Tuple

from config import GPSConfig
from reward import PaperReward
from tqdm import tqdm


class HumanoidPaperEnv:
    """
    Thin wrapper around Gymnasium MuJoCo Humanoid-v4.

    Key choices for faithfulness:
    - use full simulator state [qpos, qvel] for DDP dynamics
    - use a paper-like policy observation built from qpos/qvel
    - compute paper reward ourselves, ignoring env native reward
    """

    def __init__(self, cfg: GPSConfig):
        self.cfg = cfg
        self.env = gym.make(
            cfg.env_id,
            terminate_when_unhealthy=cfg.terminate_when_unhealthy,
            render_mode=cfg.render_mode,
        )
        self.unwrapped = self.env.unwrapped
        self.reward_fn = PaperReward(cfg.reward)

        # initialize once so dims are known
        obs, _ = self.env.reset(seed=cfg.seed)
        qpos = self.unwrapped.data.qpos.copy()
        qvel = self.unwrapped.data.qvel.copy()

        self.nq = qpos.shape[0]
        self.nv = qvel.shape[0]
        self.state_dim = self.nq + self.nv
        self.act_dim = self.env.action_space.shape[0]

        # policy obs excludes global x,y position, keeps z and all other qpos/qvel
        dummy_state = np.concatenate([qpos, qvel], axis=0)
        self.obs_dim = self.policy_obs_from_sim_state(dummy_state).shape[0]

    def reset(self, seed: int | None = None) -> np.ndarray:
        self.env.reset(seed=seed)
        return self.get_sim_state()

    def get_sim_state(self) -> np.ndarray:
        qpos = self.unwrapped.data.qpos.copy()
        qvel = self.unwrapped.data.qvel.copy()
        return np.concatenate([qpos, qvel], axis=0)

    def set_sim_state(self, sim_state: np.ndarray) -> None:
        qpos = sim_state[:self.nq].copy()
        qvel = sim_state[self.nq:].copy()
        self.unwrapped.set_state(qpos, qvel)

    def get_reward_features(self, sim_state: np.ndarray, action: np.ndarray) -> dict:
        qpos = sim_state[:self.nq]
        qvel = sim_state[self.nq:]

        # Assumption for Humanoid-v4 / MuJoCo ordering:
        # qpos[0] = root x, qpos[1] = root y, qpos[2] = root z
        # qvel[0] = root x velocity
        # This matches the usual MuJoCo free joint convention for Humanoid.
        z = float(qpos[2])
        vx = float(qvel[0])

        return {
            "vx": vx,
            "z": z,
            "u": action.copy(),
        }

    def paper_reward(self, sim_state: np.ndarray, action: np.ndarray) -> float:
        feat = self.get_reward_features(sim_state, action)
        return self.reward_fn.value(feat["vx"], feat["z"], feat["u"])

    def policy_obs_from_sim_state(self, sim_state: np.ndarray) -> np.ndarray:
        qpos = sim_state[:self.nq]
        qvel = sim_state[self.nq:]

        # Paper-like observation:
        # use joint positions/velocities, but remove absolute x,y translation
        # keep z and all other coordinates.
        qpos_obs = qpos[2:]
        return np.concatenate([qpos_obs, qvel], axis=0).astype(np.float32)

    def step_from_state(
        self, sim_state: np.ndarray, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, np.ndarray]:
        self.set_sim_state(sim_state)
        _, _, terminated, truncated, _ = self.env.step(action)

        next_state = self.get_sim_state()
        reward = self.paper_reward(sim_state, action)
        done = bool(terminated or truncated)
        obs = self.policy_obs_from_sim_state(sim_state)
        return next_state, reward, done, obs

    def rollout_open_loop(self, init_state: np.ndarray, actions: np.ndarray):
        T = actions.shape[0]
        sim_states = [init_state.copy()]
        obs = []
        rewards = []
        dones = []

        state = init_state.copy()
        for t in range(T):
            act = actions[t]
            next_state, r, d, o = self.step_from_state(state, act)
            sim_states.append(next_state.copy())
            obs.append(o.copy())
            rewards.append(r)
            dones.append(d)
            state = next_state
            if d:
                # continue with simulator state anyway for fixed horizon style
                pass

        return (
            np.asarray(sim_states, dtype=np.float64),
            np.asarray(obs, dtype=np.float32),
            actions.astype(np.float64),
            np.asarray(rewards, dtype=np.float64),
            np.asarray(dones, dtype=bool),
        )