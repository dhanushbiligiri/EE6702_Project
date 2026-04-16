import mujoco
import numpy as np


class HumanoidDynamics:
    def __init__(self, xml_path):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        self.nq = self.model.nq
        self.nv = self.model.nv
        self.nu = self.model.nu

    def f(self, x, u):
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        u = np.asarray(u, dtype=np.float64).reshape(-1)

        if x.shape[0] != self.nq + self.nv:
            raise ValueError(
                f"x must have length {self.nq + self.nv}, got {x.shape[0]}"
            )
        if u.shape[0] != self.nu:
            raise ValueError(
                f"u must have length {self.nu}, got {u.shape[0]}"
            )

        qpos = x[:self.nq]
        qvel = x[self.nq:self.nq + self.nv]

        self.data.qpos[:] = qpos
        self.data.qvel[:] = qvel
        self.data.ctrl[:] = u

        self.data.qfrc_applied[:] = 0.0
        self.data.xfrc_applied[:] = 0.0

        mujoco.mj_forward(self.model, self.data)

        h = 1e-8
        qpos_next = qpos.copy()
        mujoco.mj_integratePos(self.model, qpos_next, qvel, h)
        qdot = (qpos_next - qpos) / h

        vdot = self.data.qacc.copy()

        xdot = np.concatenate([qdot, vdot])
        return xdot