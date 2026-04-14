from __future__ import annotations
import numpy as np
from config import RewardConfig


class PaperReward:
    """
    Updated paper-style reward with uprightness term.

    r(x,u) =
        - wu * ||u||^2
        - wv * (vx - vx*)^2
        - wh * (z - z*)^2
        - wo * (upright_error)^2

    where upright_error = 1 - upright_alignment

    upright_alignment should be close to 1 when the torso/root is upright.
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

        return (
            -c.wu * float(np.dot(u, u))
            -c.wv * float((vx - c.target_vx) ** 2)
            -c.wh * float((z - c.target_z) ** 2)
            -c.wo * float(upright_error ** 2)
        )

    def derivatives_wrt_state_action(
        self,
        sim_state: np.ndarray,
        action: np.ndarray,
        nq: int,
        nv: int,
    ):
        """
        Analytic derivatives wrt full sim state s = [qpos, qvel] and action u.

        Assumptions for MuJoCo Humanoid free joint:
        - qpos[2] = root z
        - qvel[0] = root x velocity
        - qpos[3:7] = root orientation quaternion [w, x, y, z]

        Uprightness proxy:
            upright_alignment = 1 - 2*(qx^2 + qy^2)

        This is the world-z alignment of the torso's local z-axis for a unit quaternion.
        It equals 1 when upright, decreases as the torso tilts.
        """
        state_dim = nq + nv
        act_dim = action.shape[0]
        c = self.cfg

        qpos = sim_state[:nq]
        qvel = sim_state[nq:]

        z = qpos[2]
        vx = qvel[0]

        # root quaternion
        qw, qx, qy, qz = qpos[3:7]

        # upright alignment approximation
        upright_alignment = 1.0 - 2.0 * (qx * qx + qy * qy)
        upright_error = 1.0 - upright_alignment  # = 2*(qx^2 + qy^2)

        rx = np.zeros(state_dim, dtype=np.float64)
        ru = np.zeros(act_dim, dtype=np.float64)
        rxx = np.zeros((state_dim, state_dim), dtype=np.float64)
        ruu = np.zeros((act_dim, act_dim), dtype=np.float64)
        rux = np.zeros((act_dim, state_dim), dtype=np.float64)

        # ----------------------------
        # Height term: -wh (z - z*)^2
        # ----------------------------
        rx[2] += -2.0 * c.wh * (z - c.target_z)
        rxx[2, 2] += -2.0 * c.wh

        # -----------------------------------
        # Velocity term: -wv (vx - vx*)^2
        # vx is qvel[0] -> full state index nq
        # -----------------------------------
        rx[nq + 0] += -2.0 * c.wv * (vx - c.target_vx)
        rxx[nq + 0, nq + 0] += -2.0 * c.wv

        # ----------------------------------------------------
        # Upright term: -wo * (upright_error)^2
        #
        # upright_error = 2*(qx^2 + qy^2)
        #
        # Let e = 2(qx^2 + qy^2)
        # r_upright = -wo * e^2
        #
        # de/dqx = 4*qx
        # de/dqy = 4*qy
        #
        # dr/dqx = -2*wo*e*(de/dqx) = -8*wo*e*qx
        # dr/dqy = -8*wo*e*qy
        #
        # d2r/dqx2 = -8*wo*(de/dqx*qx + e*1 + qx*de/dqx)
        #          = -8*wo*(4*qx^2 + e + 4*qx^2)
        #          = -8*wo*(8*qx^2 + e)
        #
        # similarly for qy, and cross term:
        # d2r/dqxdy = -8*wo*(de/dqy*qx) = -32*wo*qx*qy
        #            plus symmetric contribution from product form already captured
        # exact expansion from r = -4wo(qx^2+qy^2)^2 gives:
        # d2r/dqx2 = -16wo*(3*qx^2 + qy^2)
        # d2r/dqy2 = -16wo*(qx^2 + 3*qy^2)
        # d2r/dqxdy = -32wo*qx*qy
        # ----------------------------------------------------
        idx_qx = 4
        idx_qy = 5

        rx[idx_qx] += -8.0 * c.wo * upright_error * qx
        rx[idx_qy] += -8.0 * c.wo * upright_error * qy

        rxx[idx_qx, idx_qx] += -16.0 * c.wo * (3.0 * qx * qx + qy * qy)
        rxx[idx_qy, idx_qy] += -16.0 * c.wo * (qx * qx + 3.0 * qy * qy)
        rxx[idx_qx, idx_qy] += -32.0 * c.wo * qx * qy
        rxx[idx_qy, idx_qx] += -32.0 * c.wo * qx * qy

        # --------------------------
        # Control term: -wu ||u||^2
        # --------------------------
        ru[:] = -2.0 * c.wu * action
        ruu[:] = -2.0 * c.wu * np.eye(act_dim)

        return rx, ru, rxx, ruu, rux