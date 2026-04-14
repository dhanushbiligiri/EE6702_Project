from __future__ import annotations
import numpy as np
import gymnasium as gym
from typing import Tuple

from config import GPSConfig
from reward import PaperReward


class HumanoidPaperEnv:
    """
    Thin wrapper around Gymnasium MuJoCo Humanoid-v4.

    Key choices:
    - use full simulator state [qpos, qvel] for DDP dynamics
    - use a paper-like policy observation built from qpos/qvel
    - compute custom reward ourselves, ignoring env native reward
    - add uprightness feature to discourage falling
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
        self.env.reset(seed=cfg.seed)
        qpos = self.unwrapped.data.qpos.copy()
        qvel = self.unwrapped.data.qvel.copy()

        self.nq = qpos.shape[0]
        self.nv = qvel.shape[0]
        self.state_dim = self.nq + self.nv
        self.act_dim = self.env.action_space.shape[0]

        dummy_state = np.concatenate([qpos, qvel], axis=0)
        self.obs_dim = self.policy_obs_from_sim_state(dummy_state).shape[0]

    def reset(self, seed: int | None = None) -> np.ndarray:
        self.env.reset(seed=seed)
        return self.get_sim_state()

    def close(self) -> None:
        self.env.close()

    def get_sim_state(self) -> np.ndarray:
        qpos = self.unwrapped.data.qpos.copy()
        qvel = self.unwrapped.data.qvel.copy()
        return np.concatenate([qpos, qvel], axis=0)

    def set_sim_state(self, sim_state: np.ndarray) -> None:
        qpos = sim_state[:self.nq].copy()
        qvel = sim_state[self.nq:].copy()
        self.unwrapped.set_state(qpos, qvel)

    def compute_upright_alignment(self, qpos: np.ndarray) -> float:
        """
        Assumes qpos[3:7] is root quaternion [w, x, y, z].

        For a unit quaternion, the world-z alignment of the body's local z-axis is:
            z_align = 1 - 2*(qx^2 + qy^2)

        z_align = 1 means upright.
        """
        qw, qx, qy, qz = qpos[3:7]
        upright_alignment = 1.0 - 2.0 * (qx * qx + qy * qy)

        # keep it numerically bounded
        upright_alignment = float(np.clip(upright_alignment, -1.0, 1.0))
        return upright_alignment

    def get_reward_features(self, sim_state: np.ndarray, action: np.ndarray) -> dict:
        qpos = sim_state[:self.nq]
        qvel = sim_state[self.nq:]

        # MuJoCo humanoid convention
        z = float(qpos[2])
        vx = float(qvel[0])
        upright_alignment = self.compute_upright_alignment(qpos)

        return {
            "vx": vx,
            "z": z,
            "upright_alignment": upright_alignment,
            "u": action.copy(),
        }

    def paper_reward(self, sim_state: np.ndarray, action: np.ndarray) -> float:
        feat = self.get_reward_features(sim_state, action)
        return self.reward_fn.value(
            vx=feat["vx"],
            z=feat["z"],
            upright_alignment=feat["upright_alignment"],
            u=feat["u"],
        )

    def policy_obs_from_sim_state(self, sim_state: np.ndarray) -> np.ndarray:
        qpos = sim_state[:self.nq]
        qvel = sim_state[self.nq:]

        # policy observation:
        # remove absolute x,y translation, keep z, orientation, joints, and velocities
        qpos_obs = qpos[2:]
        return np.concatenate([qpos_obs, qvel], axis=0).astype(np.float32)

    def step_from_state(
        self,
        sim_state: np.ndarray,
        action: np.ndarray,
    ) -> Tuple[np.ndarray, float, bool, np.ndarray]:
        self.set_sim_state(sim_state)
        _, _, terminated, truncated, _ = self.env.step(action)

        next_state = self.get_sim_state()

        # reward on current state-action
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

        return (
            np.asarray(sim_states, dtype=np.float64),
            np.asarray(obs, dtype=np.float32),
            actions.astype(np.float64),
            np.asarray(rewards, dtype=np.float64),
            np.asarray(dones, dtype=bool),
        )