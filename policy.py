from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
from tqdm import tqdm


class GaussianPolicy(nn.Module):
    """Simple Gaussian policy with:
    - one hidden layer
    - soft rectifier (Softplus)
    - linear output
    - Gaussian exploration
    """

    def __init__(self, obs_dim: int, act_dim: int, cfg):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        self.net = nn.Sequential(
            nn.Linear(obs_dim, cfg.hidden_dim),
            nn.Softplus(),
            nn.Linear(cfg.hidden_dim, act_dim),
        )
        self.log_std = nn.Parameter(torch.ones(act_dim) * cfg.init_log_std)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def forward(self, obs: torch.Tensor):
        obs = obs.to(self.device)
        mean = self.net(obs)
        log_std = torch.clamp(self.log_std, min=-6.0, max=2.0)
        std = torch.exp(log_std).expand_as(mean)
        return mean, std

    def dist(self, obs: torch.Tensor) -> Normal:
        mean, std = self.forward(obs)
        return Normal(mean, std)

    def sample_action(self, obs_np: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            obs = torch.as_tensor(
                obs_np, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            dist = self.dist(obs)
            action = dist.sample()[0].detach().cpu().numpy()
        return action

    def mean_action(self, obs_np: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            obs = torch.as_tensor(
                obs_np, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            mean, _ = self.forward(obs)
            action = mean[0].detach().cpu().numpy()
        return action

    def log_prob_actions(
        self,
        obs: torch.Tensor,       # [B, obs_dim]
        actions: torch.Tensor,   # [B, act_dim]
    ) -> torch.Tensor:
        obs = obs.to(self.device)
        actions = actions.to(self.device)
        dist = self.dist(obs)
        return dist.log_prob(actions).sum(dim=-1)

    def log_prob_prefixes_for_traj(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
    ) -> np.ndarray:
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
            act_t = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
            logps = self.log_prob_actions(obs_t, act_t).detach().cpu().numpy()
        return np.cumsum(logps)