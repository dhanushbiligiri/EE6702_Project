from __future__ import annotations
import numpy as np
import torch

from reward import PaperReward
from policy import GaussianPolicy
from env import HumanoidPaperEnv
from tqdm import tqdm


class AdaptiveReward:
    """
    Implements the paper's adaptive guide idea:
        r_bar(x,u) = r(x,u) + log pi_theta(u|x)

    For iLQR/DDP, the exact second-order derivatives of log pi wrt state/action
    are cumbersome through the neural network. The closest practical faithful choice
    is:
    - keep analytic paper reward derivatives
    - add first-order finite-difference derivatives of log pi wrt state and action
    - ignore second derivatives of the policy term

    This keeps the adaptive guide concept faithful while making the implementation
    tractable.
    """

    def __init__(
        self,
        env: HumanoidPaperEnv,
        base_reward: PaperReward,
        policy: GaussianPolicy,
        device: torch.device,
        fd_eps: float = 1e-4,
    ):
        self.env = env
        self.base_reward = base_reward
        self.policy = policy
        self.device = device
        self.fd_eps = fd_eps

    def logpi(self, sim_state: np.ndarray, action: np.ndarray) -> float:
        obs = self.env.policy_obs_from_sim_state(sim_state)
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            act_t = torch.as_tensor(action, dtype=torch.float32, device=self.device).unsqueeze(0)
            lp = self.policy.log_prob_actions(obs_t, act_t)[0].item()
        return float(lp)

    def derivatives(self, sim_state: np.ndarray, action: np.ndarray):
        rx, ru, rxx, ruu, rux = self.base_reward.derivatives_wrt_state_action(
            sim_state, action, self.env.nq, self.env.nv
        )

        state_dim = sim_state.shape[0]
        act_dim = action.shape[0]
        eps = self.fd_eps

        # first-order finite differences for log pi
        rx_pi = np.zeros(state_dim, dtype=np.float64)
        ru_pi = np.zeros(act_dim, dtype=np.float64)

        for i in range(state_dim):
            ds = np.zeros_like(sim_state)
            ds[i] = eps
            lp_p = self.logpi(sim_state + ds, action)
            lp_m = self.logpi(sim_state - ds, action)
            rx_pi[i] = (lp_p - lp_m) / (2.0 * eps)

        for j in range(act_dim):
            du = np.zeros_like(action)
            du[j] = eps
            lp_p = self.logpi(sim_state, action + du)
            lp_m = self.logpi(sim_state, action - du)
            ru_pi[j] = (lp_p - lp_m) / (2.0 * eps)

        return rx + rx_pi, ru + ru_pi, rxx, ruu, rux