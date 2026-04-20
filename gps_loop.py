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
    estimate_return_objective,
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
        self.policy_bank = {}

    def make_initial_action_guess(self, horizon: int) -> np.ndarray:
        scale = self.cfg.initial_action_guess_scale
        return scale * np.random.randn(horizon, self.env.act_dim).astype(np.float64)

    def build_initial_guides(self, init_state: np.ndarray):
        iterator = range(self.cfg.initial_num_guides)
        if self.cfg.initial_num_guides > 1:
            iterator = tqdm(iterator, desc="Initial guides", leave=False)

        guide_records = []

        for gi in iterator:
            perturbed_state = init_state.copy()
            perturbed_state[:3] += 0.01 * np.random.randn(3)
            u0 = self.make_initial_action_guess(self.cfg.ilqr.horizon)
            controller, _, r_nom = self.ilqr.optimize(
                x0=perturbed_state,
                u_init=u0,
                reward_override_fn=None,
                show_progress=True,
            )
            controller.source_name = f"guide_{gi}"
            guide_return = float(np.sum(r_nom))
            guide_records.append((guide_return, controller))
            tqdm.write(f"[Guide {gi}] return={guide_return:.4f}")

        guide_records.sort(key=lambda x: x[0], reverse=True)

        # # keep only the best guides
        num_keep = len(guide_records)
        self.controllers = [ctrl for _, ctrl in guide_records]

        tqdm.write(
            "[GPS] Keeping guides: "
            + ", ".join(ctrl.source_name for ctrl in self.controllers)
        )

    def collect_initial_guiding_samples(self, init_state: np.ndarray):
        rng = np.random.default_rng(self.cfg.seed)
        guiding_trajs = []
        if len(self.controllers) == 0:
            raise ValueError("No valid guides available for sampling.")

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
        self.policy_bank["policy_pre"] = copy.deepcopy(self.best_policy).eval()

    def lbfgs_policy_update(self, active_set, gps_iter: int):
        cache = build_sample_cache(
            trajectories=active_set,
            controllers=self.controllers,
            policy_bank=self.policy_bank,
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

    def add_onpolicy_samples(self, init_state: np.ndarray, iteration: int, source_id: str):
        for i in tqdm(
            range(self.cfg.onpolicy_samples_per_iter),
            desc=f"On-policy GPS {iteration}",
            leave=False,
        ):
            traj = rollout_policy(
                env=self.env,
                policy=self.policy,
                init_state=init_state,
                horizon=self.cfg.ilqr.horizon,
                device=self.device,
                deterministic=False,
                source_name=f"{source_id}_{i}",
                source_id=source_id,
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
        for j in tqdm(
            range(self.cfg.adaptive_guiding_samples_per_iter),
            desc=f"Adaptive samples GPS {iteration}",
            leave=False,
        ):
            traj = sample_guide_trajectory(self.env, ctrl, init_state, rng)
            traj.source_name = f"guide_adapt_{iteration}_{j}"
            traj.source_type = "guide"
            traj.source_id = ctrl.source_name
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
                source_id="policy_pre",
            )
            self.sample_set.append(traj)

        self.best_policy = copy.deepcopy(self.policy)

        initial_active_set = select_active_set(
            trajectories=self.sample_set,
            controllers=self.controllers,
            policy_bank=self.policy_bank,
            current_best_policy=self.best_policy,
            active_set_size=self.cfg.active_set_size,
            device=self.device,
        )

        initial_cache = build_sample_cache(
            trajectories=initial_active_set,
            controllers=self.controllers,
            policy_bank=self.policy_bank,
        )

        self.best_return = estimate_return_objective(
            policy=self.best_policy,
            cache=initial_cache,
            device=self.device,
        )

        history = []

        gps_bar = tqdm(range(1, self.cfg.gps_iterations + 1), desc="GPS Iterations")
        for k in gps_bar:
            tqdm.write(f"\n===== GPS ITERATION {k} =====")
            tqdm.write(f"\n[GPS {k}] selecting active set")
            active_set = select_active_set(
                trajectories=self.sample_set,
                controllers=self.controllers,
                policy_bank=self.policy_bank,
                current_best_policy=self.best_policy,
                active_set_size=self.cfg.active_set_size,
                device=self.device,
            )

            self.policy.load_state_dict(self.best_policy.state_dict())
            self.lbfgs_policy_update(active_set, gps_iter=k)

            candidate_id = f"policy_{k}"
            self.policy_bank[candidate_id] = copy.deepcopy(self.policy).eval()

            self.add_onpolicy_samples(
                init_state=init_state,
                iteration=k,
                source_id=candidate_id,
            )

            if self.cfg.use_adaptive_guides and k <= self.cfg.adaptive_until_iter:
                self.add_adaptive_guides(init_state, iteration=k)

            comparison_set = select_active_set(
                trajectories=self.sample_set,
                controllers=self.controllers,
                policy_bank=self.policy_bank,
                current_best_policy=self.best_policy,
                active_set_size=self.cfg.active_set_size,
                device=self.device,
            )

            comparison_cache = build_sample_cache(
                trajectories=comparison_set,
                controllers=self.controllers,
                policy_bank=self.policy_bank,
            )

            candidate_est = estimate_return_objective(
                policy=self.policy,
                cache=comparison_cache,
                device=self.device,
            )

            best_est = estimate_return_objective(
                policy=self.best_policy,
                cache=comparison_cache,
                device=self.device,
            )

            improved = candidate_est > best_est

            if improved:
                self.best_return = candidate_est
                self.best_policy = copy.deepcopy(self.policy)
                self.wr = max(self.cfg.wr_min, self.wr / 1.2)
            else:
                self.policy.load_state_dict(self.best_policy.state_dict())
                self.wr = min(self.cfg.wr_max, self.wr * 1.2)

                if self.cfg.resample_best_on_reject:
                    best_id = f"best_resample_{k}"
                    self.policy_bank[best_id] = copy.deepcopy(self.best_policy).eval()

                    for i in range(self.cfg.best_policy_resample_count):
                        traj = rollout_policy(
                            env=self.env,
                            policy=self.best_policy,
                            init_state=init_state,
                            horizon=self.cfg.ilqr.horizon,
                            device=self.device,
                            deterministic=False,
                            source_name=f"{best_id}_{i}",
                            source_id=best_id,
                        )
                        self.sample_set.append(traj)

            debug_eval = self.estimate_current_policy_value(init_state)

            record = {
                "iter": k,
                "candidate_est": candidate_est,
                "best_est": best_est,
                "debug_eval_return": debug_eval,
                "best_return": self.best_return,
                "wr": self.wr,
                "num_samples": len(self.sample_set),
                "num_controllers": len(self.controllers),
            }
            history.append(record)

            gps_bar.set_postfix(
                cand=f"{candidate_est:.2f}",
                best=f"{best_est:.2f}",
                eval=f"{debug_eval:.2f}",
                wr=f"{self.wr:.1e}",
                samples=len(self.sample_set),
                ctrls=len(self.controllers),
            )

            tqdm.write(
                f"[GPS {k}] cand_est={candidate_est:.2f} | best_est={best_est:.2f} "
                f"| eval={debug_eval:.2f} | wr={self.wr:.2e} "
                f"| samples={len(self.sample_set)} | ctrls={len(self.controllers)}"
            )

        return {
            "policy": self.best_policy,
            "history": history,
            "best_return": self.best_return,
            "controllers": self.controllers,
            "sample_set": self.sample_set,
        }