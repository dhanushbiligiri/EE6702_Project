from __future__ import annotations
import copy
import numpy as np
import torch
from tqdm import tqdm

from config import GPSConfig
from utils import set_seed
from env import HumanoidPaperEnv
from reward import PaperReward
from dynamics import FiniteDifferenceDynamics
from ilqr import ILQR
from guide import sample_guide_trajectory
from policy import GaussianPolicy
from pretrain import pretrain_policy_from_guides
from importance_objective import (
    build_sample_cache,
    phi_objective,
    select_active_set,
)
from evaluate import rollout_policy, evaluate_policy
from adaptive import AdaptiveReward


class GuidedPolicySearchTrainer:
    def __init__(self, cfg: GPSConfig, device: str = "cpu"):
        self.cfg = cfg
        self.device = torch.device(device)
        set_seed(cfg.seed)

        self.env = HumanoidPaperEnv(cfg)
        self.reward = PaperReward(cfg.reward)
        self.dyn = FiniteDifferenceDynamics(self.env, fd_eps=cfg.ilqr.fd_eps)
        self.ilqr = ILQR(self.env, self.reward, self.dyn, cfg.ilqr)

        self.policy = GaussianPolicy(
            obs_dim=self.env.obs_dim,
            act_dim=self.env.act_dim,
            cfg=cfg.policy,
        ).to(self.device)

        self.best_policy = copy.deepcopy(self.policy)
        self.best_return = -np.inf
        self.wr = cfg.wr_init

        self.controllers = []
        self.sample_set = []

    def make_initial_action_guess(self, horizon: int) -> np.ndarray:
        return 0.05 * np.random.randn(horizon, self.env.act_dim).astype(np.float64)

    def build_initial_guides(self, init_state: np.ndarray):
        iterator = range(self.cfg.initial_num_guides)
        if self.cfg.initial_num_guides > 1:
            iterator = tqdm(iterator, desc="Initial guides", leave=False)

        for gi in iterator:
            u0 = self.make_initial_action_guess(self.cfg.ilqr.horizon)
            controller, _, _ = self.ilqr.optimize(
                x0=init_state,
                u_init=u0,
                reward_override_fn=None,
                show_progress=True,
            )
            controller.source_name = f"guide_{gi}"
            self.controllers.append(controller)

    def collect_initial_guiding_samples(self, init_state: np.ndarray):
        rng = np.random.default_rng(self.cfg.seed)
        guiding_trajs = []

        per_guide = max(1, self.cfg.initial_guiding_samples // len(self.controllers))
        for ctrl in tqdm(self.controllers, desc="Guide sampling", leave=False):
            for _ in tqdm(range(per_guide), desc=f"Sampling {ctrl.source_name}", leave=False):
                traj = sample_guide_trajectory(self.env, ctrl, init_state, rng)
                guiding_trajs.append(traj)

        self.sample_set.extend(guiding_trajs)
        return guiding_trajs

    def pretrain_from_guides(self, guiding_trajs):
        obs = np.concatenate([tr.obs for tr in guiding_trajs], axis=0)
        acts = np.concatenate([tr.actions for tr in guiding_trajs], axis=0)
        pretrain_policy_from_guides(
            policy=self.policy,
            guide_obs=obs,
            guide_actions=acts,
            epochs=self.cfg.pretrain_epochs,
            batch_size=self.cfg.pretrain_batch_size,
            device=str(self.device),
            show_progress=True,
        )
        self.best_policy = copy.deepcopy(self.policy)

    def lbfgs_policy_update(self, active_set, gps_iter: int):
        cache = build_sample_cache(
            trajectories=active_set,
            controllers=self.controllers,
            policy=self.policy,
            device=self.device,
        )

        optimizer = torch.optim.LBFGS(
            self.policy.parameters(),
            lr=1.0,
            max_iter=self.cfg.lbfgs_max_iter,
            history_size=self.cfg.lbfgs_history_size,
            line_search_fn="strong_wolfe",
        )

        closure_calls = {"count": 0}

        def closure():
            optimizer.zero_grad()
            phi = phi_objective(
                policy=self.policy,
                cache=cache,
                wr=self.wr,
                device=self.device,
            )
            loss = -phi
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 10.0)

            closure_calls["count"] += 1
            tqdm.write(
                f"[LBFGS | GPS {gps_iter}] closure={closure_calls['count']} | phi={phi.item():.6f} | loss={loss.item():.6f}"
            )
            return loss

        tqdm.write(f"[LBFGS | GPS {gps_iter}] starting update on active_set={len(active_set)}")
        optimizer.step(closure)
        tqdm.write(f"[LBFGS | GPS {gps_iter}] done")

    def add_onpolicy_samples(self, init_state: np.ndarray, iteration: int):
        for i in tqdm(range(self.cfg.onpolicy_samples_per_iter), desc=f"On-policy GPS {iteration}", leave=False):
            traj = rollout_policy(
                env=self.env,
                policy=self.policy,
                init_state=init_state,
                horizon=self.cfg.ilqr.horizon,
                device=self.device,
                deterministic=False,
                source_name=f"policy_{iteration}_{i}",
            )
            self.sample_set.append(traj)

    def add_adaptive_guides(self, init_state: np.ndarray, iteration: int):
        adaptive_reward = AdaptiveReward(
            env=self.env,
            base_reward=self.reward,
            policy=self.policy,
            device=self.device,
            fd_eps=self.cfg.ilqr.fd_eps,
        )

        u_seed = self.make_initial_action_guess(self.cfg.ilqr.horizon)
        tqdm.write(f"[Adaptive Guide | GPS {iteration}] optimizing adaptive guide")
        ctrl, _, _ = self.ilqr.optimize(
            x0=init_state,
            u_init=u_seed,
            reward_override_fn=adaptive_reward.derivatives,
            show_progress=True,
        )
        ctrl.source_name = f"guide_adapt_{iteration}"
        self.controllers.append(ctrl)

        rng = np.random.default_rng(self.cfg.seed + iteration + 1000)
        for j in tqdm(range(self.cfg.adaptive_guiding_samples_per_iter), desc=f"Adaptive samples GPS {iteration}", leave=False):
            traj = sample_guide_trajectory(self.env, ctrl, init_state, rng)
            traj.source_name = f"guide_adapt_{iteration}_{j}"
            self.sample_set.append(traj)

    def estimate_current_policy_value(self, init_state: np.ndarray) -> float:
        out = evaluate_policy(
            env=self.env,
            policy=self.policy,
            init_state=init_state,
            horizon=self.cfg.ilqr.horizon,
            device=self.device,
            n_rollouts=3,
            deterministic=True,
        )
        return out["mean_return"]

    def train(self):
        init_state = self.env.reset(seed=self.cfg.seed)

        tqdm.write("[GPS] Building initial guides")
        self.build_initial_guides(init_state)

        tqdm.write("[GPS] Collecting initial guiding samples")
        guiding_trajs = self.collect_initial_guiding_samples(init_state)

        tqdm.write("[GPS] Pretraining policy from guides")
        self.pretrain_from_guides(guiding_trajs)

        tqdm.write("[GPS] Collecting initial on-policy samples")
        for i in tqdm(range(self.cfg.onpolicy_samples_per_iter), desc="Initial policy rollouts", leave=False):
            traj = rollout_policy(
                env=self.env,
                policy=self.policy,
                init_state=init_state,
                horizon=self.cfg.ilqr.horizon,
                device=self.device,
                deterministic=False,
                source_name=f"policy_pre_{i}",
            )
            self.sample_set.append(traj)

        self.best_policy = copy.deepcopy(self.policy)
        self.best_return = self.estimate_current_policy_value(init_state)

        history = []

        gps_bar = tqdm(range(1, self.cfg.gps_iterations + 1), desc="GPS Iterations")
        for k in gps_bar:
            tqdm.write(f"\n[GPS {k}] selecting active set")
            active_set = select_active_set(
                trajectories=self.sample_set,
                controllers=self.controllers,
                current_best_policy=self.best_policy,
                active_set_size=self.cfg.active_set_size,
                device=self.device,
            )

            self.lbfgs_policy_update(active_set, gps_iter=k)
            self.add_onpolicy_samples(init_state, iteration=k)
            self.add_adaptive_guides(init_state, iteration=k)

            current_return = self.estimate_current_policy_value(init_state)

            improved = current_return > self.best_return
            if improved:
                self.best_return = current_return
                self.best_policy = copy.deepcopy(self.policy)
                self.wr = max(self.cfg.wr_min, self.wr / 10.0)
            else:
                self.wr = min(self.cfg.wr_max, self.wr * 10.0)

            record = {
                "iter": k,
                "mean_return": current_return,
                "best_return": self.best_return,
                "wr": self.wr,
                "num_samples": len(self.sample_set),
                "num_controllers": len(self.controllers),
            }
            history.append(record)

            gps_bar.set_postfix(
                mean_return=f"{current_return:.2f}",
                best_return=f"{self.best_return:.2f}",
                wr=f"{self.wr:.1e}",
                samples=len(self.sample_set),
                ctrls=len(self.controllers),
            )

            tqdm.write(
                f"[GPS {k}] return={current_return:.2f} | best={self.best_return:.2f} | wr={self.wr:.2e} | samples={len(self.sample_set)} | ctrls={len(self.controllers)}"
            )

        return {
            "policy": self.best_policy,
            "history": history,
            "best_return": self.best_return,
            "controllers": self.controllers,
            "sample_set": self.sample_set,
        }