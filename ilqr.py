from __future__ import annotations
import numpy as np
from tqdm import tqdm

from trajectory import LinearGaussianController
from utils import ensure_psd, stable_inv_and_logdet


class ILQR:
    """
    iLQR / DDP-style local optimizer for deterministic simulator rollouts.

    Notes:
    - uses finite-difference dynamics Jacobians
    - uses analytic reward derivatives
    - builds a local linear feedback law
    - forms Gaussian guide covariance as Sigma_t = -(Q_uu)^(-1)

    Since we maximize reward, Q_uu should be negative definite near a local maximum.
    """

    def __init__(self, env, reward, dyn, cfg):
        self.env = env
        self.reward = reward
        self.dyn = dyn
        self.cfg = cfg

    def rollout_with_feedback(
        self,
        x0: np.ndarray,
        x_nom: np.ndarray,
        u_nom: np.ndarray,
        k: np.ndarray,
        K: np.ndarray,
        alpha: float = 1.0,
    ):
        T = u_nom.shape[0]
        state_dim = x0.shape[0]
        act_dim = u_nom.shape[1]

        xs = np.zeros((T + 1, state_dim), dtype=np.float64)
        us = np.zeros((T, act_dim), dtype=np.float64)
        rs = np.zeros(T, dtype=np.float64)
        ds = np.zeros(T, dtype=bool)
        obs = np.zeros((T, self.env.obs_dim), dtype=np.float32)

        xs[0] = x0.copy()

        for t in range(T):
            dx = xs[t] - x_nom[t]
            u = u_nom[t] + alpha * k[t] + K[t] @ dx
            u = np.clip(u, self.env.env.action_space.low, self.env.env.action_space.high)
            us[t] = u

            x_next, r, d, o = self.env.step_from_state(xs[t], u)
            xs[t + 1] = x_next
            rs[t] = r
            ds[t] = d
            obs[t] = o

        return xs, obs, us, rs, ds

    def total_return(self, rewards: np.ndarray) -> float:
        return float(np.sum(rewards))

    @staticmethod
    def _symmetrize(mat: np.ndarray) -> np.ndarray:
        return 0.5 * (mat + mat.T)

    def _spectral_clip_symmetric(self, mat: np.ndarray, clip_value: float) -> np.ndarray:
        """
        Preserve symmetry and clip eigenvalues, instead of elementwise clipping.
        """
        mat = self._symmetrize(mat)
        eigvals, eigvecs = np.linalg.eigh(mat)
        eigvals = np.clip(eigvals, -clip_value, clip_value)
        clipped = eigvecs @ np.diag(eigvals) @ eigvecs.T
        return self._symmetrize(clipped)

    def _check_explosion(self, name: str, mat: np.ndarray, t: int, threshold: float = 1e12):
        if np.any(~np.isfinite(mat)):
            raise np.linalg.LinAlgError(f"{name} has non-finite values at t={t}")
        max_abs = np.max(np.abs(mat))
        if max_abs > threshold:
            raise np.linalg.LinAlgError(
                f"{name} exploded at t={t}, max_abs={max_abs:.3e}"
            )

    def backward_pass(
        self,
        x_nom: np.ndarray,
        u_nom: np.ndarray,
        fx: np.ndarray,
        fu: np.ndarray,
        reg: float,
        reward_override_fn=None,
        debug: bool = False,
    ):
        T = u_nom.shape[0]
        state_dim = x_nom.shape[1]
        act_dim = u_nom.shape[1]

        # terminal value approximation
        Vx = np.zeros(state_dim, dtype=np.float64)
        Vxx = np.zeros((state_dim, state_dim), dtype=np.float64)

        k = np.zeros((T, act_dim), dtype=np.float64)
        K = np.zeros((T, act_dim, state_dim), dtype=np.float64)
        cov = np.zeros((T, act_dim, act_dim), dtype=np.float64)
        cov_inv = np.zeros((T, act_dim, act_dim), dtype=np.float64)
        cov_logdet = np.zeros(T, dtype=np.float64)

        # configurable clip from cfg if present, else fallback
        matrix_clip = getattr(self.cfg, "matrix_clip", 1e6)

        for t in reversed(range(T)):
            s = x_nom[t]
            u = u_nom[t]

            if reward_override_fn is None:
                rx, ru, rxx, ruu, rux = self.reward.derivatives_wrt_state_action(
                    s, u, self.env.nq, self.env.nv
                )
            else:
                rx, ru, rxx, ruu, rux = reward_override_fn(s, u)

            fx_t = fx[t]
            fu_t = fu[t]

            # local quadratic expansion terms
            Qx = rx + fx_t.T @ Vx
            Qu = ru + fu_t.T @ Vx
            Qxx = rxx + fx_t.T @ Vxx @ fx_t
            Quu = ruu + fu_t.T @ Vxx @ fu_t
            Qux = rux + fu_t.T @ Vxx @ fx_t

            # preserve symmetry
            Qxx = self._symmetrize(Qxx)
            Quu = self._symmetrize(Quu)

            # hard guards before things become nonsense
            self._check_explosion("Qxx", Qxx, t)
            self._check_explosion("Quu", Quu, t)
            self._check_explosion("Qux", Qux, t)

            # spectral clipping, safer than elementwise clipping
            Qxx = self._spectral_clip_symmetric(Qxx, matrix_clip)
            Quu = self._spectral_clip_symmetric(Quu, matrix_clip)
            Qux = np.clip(Qux, -matrix_clip, matrix_clip)

            # optional sparse debug printing only at a few timesteps
            if debug and t in {T - 1, T // 2, 0}:
                eigvals = np.linalg.eigvalsh(Quu)
                print(
                    f"[backward t={t}] min_eig(Quu)={eigvals.min():.6e}, "
                    f"max_eig(Quu)={eigvals.max():.6e}, reg={reg:.2e}"
                )

            # reward maximization form: push Quu more negative
            Quu_reg = Quu - reg * np.eye(act_dim, dtype=np.float64)
            Quu_reg = self._symmetrize(Quu_reg)

            minus_Quu = -Quu_reg

            if debug and t in {T - 1, T // 2, 0}:
                eigvals_m = np.linalg.eigvalsh(self._symmetrize(minus_Quu))
                print(
                    f"[backward t={t}] min_eig(-Quu_reg)={eigvals_m.min():.6e}, "
                    f"max_eig(-Quu_reg)={eigvals_m.max():.6e}"
                )

            # ensure PSD for covariance / solve
            try:
                minus_Quu = ensure_psd(minus_Quu, jitter=self.cfg.q_uu_jitter)
                minus_Quu_inv, minus_Quu_logdet = stable_inv_and_logdet(
                    minus_Quu, jitter=self.cfg.q_uu_jitter
                )

                # Since k = -Quu^{-1} Qu and Quu^{-1} = -( -Quu )^{-1}
                k_t = minus_Quu_inv @ Qu
                K_t = minus_Quu_inv @ Qux

                # Gaussian covariance Sigma = -Quu^{-1} = ( -Quu )^{-1}
                Sigma_t = minus_Quu_inv
                Sigma_inv_t = minus_Quu
                Sigma_logdet_t = minus_Quu_logdet

            except np.linalg.LinAlgError as exc:
                raise np.linalg.LinAlgError(
                    f"Backward pass failed at t={t}: {exc}"
                ) from exc

            k[t] = k_t
            K[t] = K_t
            cov[t] = Sigma_t
            cov_inv[t] = Sigma_inv_t
            cov_logdet[t] = Sigma_logdet_t

            # IMPORTANT: use Quu_reg consistently in value recursion
            Vx = Qx + K_t.T @ Quu_reg @ k_t - K_t.T @ Qu - Qux.T @ k_t
            Vxx = Qxx + K_t.T @ Quu_reg @ K_t - K_t.T @ Qux - Qux.T @ K_t
            Vxx = self._symmetrize(Vxx)

            self._check_explosion("Vxx", Vxx, t)
            Vxx = self._spectral_clip_symmetric(Vxx, matrix_clip)

        return k, K, cov, cov_inv, cov_logdet

    def optimize(
        self,
        x0: np.ndarray,
        u_init: np.ndarray,
        reward_override_fn=None,
        show_progress: bool = True,
        debug_backward: bool = False,
    ):
        T = u_init.shape[0]
        reg = self.cfg.reg_init

        x_nom, obs_nom, u_nom, r_nom, d_nom = self.rollout_with_feedback(
            x0=x0,
            x_nom=np.vstack([x0[None], np.repeat(x0[None], T, axis=0)]),
            u_nom=u_init.copy(),
            k=np.zeros_like(u_init),
            K=np.zeros((T, u_init.shape[1], x0.shape[0])),
            alpha=0.0,
        )
        best_return = self.total_return(r_nom)

        act_dim = u_init.shape[1]
        state_dim = x0.shape[0]

        # Safe fallback controller in case no improving step is found
        k_best = np.zeros((T, act_dim), dtype=np.float64)
        K_best = np.zeros((T, act_dim, state_dim), dtype=np.float64)
        cov_best = np.stack(
            [np.eye(act_dim, dtype=np.float64) * 1e-2 for _ in range(T)],
            axis=0,
        )
        cov_inv_best = np.stack(
            [np.eye(act_dim, dtype=np.float64) * 1e2 for _ in range(T)],
            axis=0,
        )
        cov_logdet_best = np.full(T, act_dim * np.log(1e-2), dtype=np.float64)

        iterator = range(self.cfg.max_iter)
        if show_progress:
            iterator = tqdm(iterator, desc="iLQR", leave=False)

        for i in iterator:
            if show_progress:
                tqdm.write(
                    f"[iLQR] iter={i} | current_best_return={best_return:.4f} | reg={reg:.2e}"
                )

            fx, fu, _ = self.dyn.linearize_trajectory(x_nom[:-1], u_nom)

            success = False
            while reg <= self.cfg.reg_max:
                try:
                    k, K, cov, cov_inv, cov_logdet = self.backward_pass(
                        x_nom=x_nom[:-1],
                        u_nom=u_nom,
                        fx=fx,
                        fu=fu,
                        reg=reg,
                        reward_override_fn=reward_override_fn,
                        debug=debug_backward,
                    )
                except np.linalg.LinAlgError as exc:
                    reg *= self.cfg.reg_scale_up
                    if show_progress:
                        tqdm.write(
                            f"  backward pass failed ({exc}), increasing reg -> {reg:.2e}"
                        )
                    continue

                # latest feasible controller as fallback
                k_best = k.copy()
                K_best = K.copy()
                cov_best = cov.copy()
                cov_inv_best = cov_inv.copy()
                cov_logdet_best = cov_logdet.copy()

                improved = False
                best_trial = None

                for alpha in self.cfg.line_search_alphas:
                    xs, obs, us, rs, ds = self.rollout_with_feedback(
                        x0=x0,
                        x_nom=x_nom[:-1],
                        u_nom=u_nom,
                        k=k,
                        K=K,
                        alpha=alpha,
                    )
                    ret = self.total_return(rs)

                    if show_progress:
                        tqdm.write(f"  alpha={alpha:.3f} | trial_return={ret:.4f}")

                    if ret > best_return:
                        improved = True
                        best_return = ret
                        best_trial = (xs, obs, us, rs, ds, k, K, cov, cov_inv, cov_logdet)

                if improved and best_trial is not None:
                    (
                        x_nom,
                        obs_nom,
                        u_nom,
                        r_nom,
                        d_nom,
                        k_best,
                        K_best,
                        cov_best,
                        cov_inv_best,
                        cov_logdet_best,
                    ) = best_trial
                    reg = max(self.cfg.reg_min, reg * self.cfg.reg_scale_down)
                    if show_progress:
                        tqdm.write(
                            f"  accepted step | new_best_return={best_return:.4f} | reg={reg:.2e}"
                        )
                    success = True
                    break
                else:
                    reg *= self.cfg.reg_scale_up
                    if show_progress:
                        tqdm.write(f"  no improvement, increasing reg -> {reg:.2e}")

            if not success:
                if show_progress:
                    tqdm.write("[iLQR] stopping: no successful step found")
                break

        controller = LinearGaussianController(
            x_nom=x_nom,
            u_nom=u_nom,
            k=k_best,
            K=K_best,
            cov=cov_best,
            cov_inv=cov_inv_best,
            cov_logdet=cov_logdet_best,
            source_name="guide",
        )
        return controller, obs_nom, r_nom