from dataclasses import dataclass
from typing import List, Optional
import numpy as np
from tqdm import tqdm


@dataclass
class TimeStepData:
    sim_state: np.ndarray      # full state [qpos, qvel]
    obs: np.ndarray            # policy observation
    action: np.ndarray
    reward: float
    next_sim_state: np.ndarray
    done: bool


@dataclass
class Trajectory:
    sim_states: np.ndarray
    obs: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray

    source_name: str
    source_type: str            # policy
    source_id: str


@dataclass
class LinearGaussianController:
    x_nom: np.ndarray          # [T+1, state_dim]
    u_nom: np.ndarray          # [T, act_dim]
    k: np.ndarray              # [T, act_dim]
    K: np.ndarray              # [T, act_dim, state_dim]
    cov: np.ndarray            # [T, act_dim, act_dim]
    cov_inv: np.ndarray        # [T, act_dim, act_dim]
    cov_logdet: np.ndarray     # [T]
    source_name: str = "guide"