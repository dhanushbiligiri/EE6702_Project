from dataclasses import dataclass, field
from typing import Tuple, Optional
from tqdm import tqdm


@dataclass
class RewardConfig:
    # Paper-style locomotion reward:
    # r = -wu ||u||^2 - wv (vx - vx*)^2 - wh (z - z*)^2
    wu: float = 0.01
    wv: float = 1.0
    wh: float = 5.0
    target_vx: float = 1.5
    target_z: float = 1.4


@dataclass
class ILQRConfig:
    horizon: int = 300
    max_iter: int = 30
    fd_eps: float = 1e-4
    reg_min: float = 1e-6
    reg_max: float = 1e6
    reg_init: float = 1.0
    reg_scale_up: float = 10.0
    reg_scale_down: float = 0.3
    line_search_alphas: Tuple[float, ...] = (
        1.0, 0.5, 0.25, 0.1, 0.05, 0.01
    )
    q_uu_jitter: float = 1e-6


@dataclass
class PolicyConfig:
    hidden_dim: int = 200
    init_log_std: float = -2.0


@dataclass
class GPSConfig:
    env_id: str = "Humanoid-v4"
    seed: int = 42

    # Paper-like training structure
    gps_iterations: int = 30
    initial_num_guides: int = 1
    initial_guiding_samples: int = 20
    onpolicy_samples_per_iter: int = 5
    adaptive_guiding_samples_per_iter: int = 5
    active_set_size: int = 40

    # Pretraining and policy optimization
    pretrain_epochs: int = 30
    pretrain_batch_size: int = 256
    lbfgs_max_iter: int = 25
    lbfgs_history_size: int = 20

    # Importance objective regularization
    wr_init: float = 1e-4
    wr_min: float = 1e-6
    wr_max: float = 1e-2

    # Humanoid wrapper assumptions
    terminate_when_unhealthy: bool = False
    render_mode: Optional[str] = None

    reward: RewardConfig = field(default_factory=RewardConfig)
    ilqr: ILQRConfig = field(default_factory=ILQRConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)