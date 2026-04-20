from __future__ import annotations
import numpy as np
from config import RewardConfig


class PaperReward:
    """
    r(x,u) =
        - wu * ||u||^2
        - wv * (vx - vx*)^2
        - wh * (z - z*)^2
        - wo * (upright_error)^2

    where:
        upright_error = 1 - upright_alignment
        upright_alignment = 1 - 2*(qx^2 + qy^2)

    """

    def __init__(self, cfg: RewardConfig):
        self.cfg = cfg

    def value(
        self,
        vx: float,
        z: float,
        upright_alignment: float,
        u: np.ndarray,
    ) -> float:
        c = self.cfg

        upright_error = 1.0 - upright_alignment

        reward = (
            -c.wu * float(np.dot(u, u))
            -c.wv * float((vx - c.target_vx) ** 2)
            -c.wh * float((z - c.target_z) ** 2)
            -c.wo * float(upright_error ** 2)
        )

        return reward

    def derivatives_wrt_state_action(
        self,
        sim_state: np.ndarray,
        action: np.ndarray,
        nq: int,
        nv: int,
    ):
        """
        Assumptions for MuJoCo Humanoid free joint:
        - qpos[2] = root z
        - qvel[0] = root x velocity
        - qpos[3:7] = root orientation quaternion [w, x, y, z]

        Uprightness proxy:
            upright_alignment = 1 - 2*(qx^2 + qy^2)
            upright_error = 1 - upright_alignment = 2*(qx^2 + qy^2)
        """
        state_dim = nq + nv
        act_dim = action.shape[0]
        c = self.cfg

        qpos = sim_state[:nq]
        qvel = sim_state[nq:]

        z = qpos[2]
        vx = qvel[0]
        # quaternion
        _, qx, qy, _ = qpos[3:7]

        # upright alignment
        upright_alignment = 1.0 - 2.0 * (qx * qx + qy * qy)
        upright_error = 1.0 - upright_alignment

        rx = np.zeros(state_dim, dtype=np.float64)
        ru = np.zeros(act_dim, dtype=np.float64)
        rxx = np.zeros((state_dim, state_dim), dtype=np.float64)
        ruu = np.zeros((act_dim, act_dim), dtype=np.float64)
        rux = np.zeros((act_dim, state_dim), dtype=np.float64)

        # Height term
        rx[2] += -2.0 * c.wh * (z - c.target_z)
        rxx[2, 2] += -2.0 * c.wh

        # Velocity term
        rx[nq + 0] += -2.0 * c.wv * (vx - c.target_vx)
        rxx[nq + 0, nq + 0] += -2.0 * c.wv

        # Upright term
        idx_qx = 4
        idx_qy = 5

        rx[idx_qx] += -8.0 * c.wo * upright_error * qx
        rx[idx_qy] += -8.0 * c.wo * upright_error * qy

        rxx[idx_qx, idx_qx] += -16.0 * c.wo * (3.0 * qx * qx + qy * qy)
        rxx[idx_qy, idx_qy] += -16.0 * c.wo * (qx * qx + 3.0 * qy * qy)
        rxx[idx_qx, idx_qy] += -32.0 * c.wo * qx * qy
        rxx[idx_qy, idx_qx] += -32.0 * c.wo * qx * qy

        # Control term
        ru[:] = -2.0 * c.wu * action
        ruu[:] = -2.0 * c.wu * np.eye(act_dim, dtype=np.float64)

        return rx, ru, rxx, ruu, rux