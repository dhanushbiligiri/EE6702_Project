from dataclasses import dataclass, field
from typing import Tuple, Optional

@dataclass
class RewardConfig:
    wu: float = 0.005
    wv: float = 20.0
    wh: float = 6.0
    wo: float = 6.0

    target_vx: float = 0.4
    target_z: float = 1.4


@dataclass
class ILQRConfig:
    horizon: int = 500
    max_iter: int = 15
    fd_eps: float = 1e-3
    reg_min: float = 1e-6
    reg_max: float = 1e8
    reg_init: float = 1e4
    reg_scale_up: float = 10.0
    reg_scale_down: float = 0.5
    line_search_alphas: Tuple[float, ...] = (1.0, 0.5, 0.25, 0.1, 0.05)
    q_uu_jitter: float = 1e-6
    matrix_clip: float = 1e6

    # NE
    guide_cov_max_eig: float = 0.3
    guide_cov_min_eig: float = 1e-6


@dataclass
class PolicyConfig:
    hidden_dim: int = 200
    init_log_std: float = -1.0 

@dataclass
class GPSConfig:
    env_id: str = "Humanoid-v4"
    seed: int = 42

    gps_iterations: int = 25

    initial_num_guides: int = 4
    initial_guiding_samples: int = 40
    onpolicy_samples_per_iter: int = 25
    adaptive_guiding_samples_per_iter: int = 10
    active_set_size: int = 50

    pretrain_epochs: int = 30
    pretrain_batch_size: int = 256
    lbfgs_max_iter: int = 25
    lbfgs_history_size: int = 20

    wr_init: float = 1e-3
    wr_min: float = 1e-6
    wr_max: float = 1e-1

    terminate_when_unhealthy: bool = True
    render_mode: Optional[str] = None

    use_adaptive_guides: bool = False
    adaptive_until_iter: int = 60
    resample_best_on_reject: bool = True
    best_policy_resample_count: int = 5
    initial_action_guess_scale: float = 0.3

    reward: RewardConfig = field(default_factory=RewardConfig)
    ilqr: ILQRConfig = field(default_factory=ILQRConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
