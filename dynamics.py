from __future__ import annotations
import numpy as np
from env import HumanoidPaperEnv
from tqdm import tqdm


class FiniteDifferenceDynamics:
    def __init__(self, env: HumanoidPaperEnv, fd_eps: float = 1e-4):
        self.env = env
        self.fd_eps = fd_eps

    def step_fn(self, sim_state: np.ndarray, action: np.ndarray) -> np.ndarray:
        next_state, _, _, _ = self.env.step_from_state(sim_state, action)
        return next_state

    def linearize_step(self, sim_state: np.ndarray, action: np.ndarray):
        state_dim = sim_state.shape[0]
        act_dim = action.shape[0]
        eps = self.fd_eps

        f0 = self.step_fn(sim_state, action)

        fx = np.zeros((state_dim, state_dim), dtype=np.float64)
        fu = np.zeros((state_dim, act_dim), dtype=np.float64)

        for i in range(state_dim):
            ds = np.zeros_like(sim_state)
            ds[i] = eps
            fp = self.step_fn(sim_state + ds, action)
            fm = self.step_fn(sim_state - ds, action)
            fx[:, i] = (fp - fm) / (2.0 * eps)

        for j in range(act_dim):
            du = np.zeros_like(action)
            du[j] = eps
            fp = self.step_fn(sim_state, action + du)
            fm = self.step_fn(sim_state, action - du)
            fu[:, j] = (fp - fm) / (2.0 * eps)

        return fx, fu, f0

    def linearize_trajectory(
        self,
        sim_states: np.ndarray,
        actions: np.ndarray,
    ):
        T = actions.shape[0]
        state_dim = sim_states.shape[1]
        act_dim = actions.shape[1]

        fx = np.zeros((T, state_dim, state_dim), dtype=np.float64)
        fu = np.zeros((T, state_dim, act_dim), dtype=np.float64)
        f_next = np.zeros((T, state_dim), dtype=np.float64)

        for t in range(T):
            fx_t, fu_t, f0_t = self.linearize_step(sim_states[t], actions[t])
            fx[t] = fx_t
            fu[t] = fu_t
            f_next[t] = f0_t

        return fx, fu, f_next