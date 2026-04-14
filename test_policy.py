from __future__ import annotations
import time
import numpy as np
import torch

from config import GPSConfig
from env import HumanoidPaperEnv
from policy import GaussianPolicy


def rollout_render(
    env: HumanoidPaperEnv,
    policy: GaussianPolicy,
    horizon: int,
    deterministic: bool = True,
    sleep: float = 0.03,
):
    state = env.reset(seed=env.cfg.seed)

    total_return = 0.0
    heights = []
    vxs = []
    action_norms = []

    for t in range(horizon):
        obs = env.policy_obs_from_sim_state(state)

        if deterministic:
            action = policy.mean_action(obs)
        else:
            action = policy.sample_action(obs)

        action = np.clip(action, env.env.action_space.low, env.env.action_space.high)

        feat = env.get_reward_features(state, action)
        heights.append(feat["z"])
        vxs.append(feat["vx"])
        action_norms.append(float(np.linalg.norm(action)))
        upright = feat["upright_alignment"]

        next_state, reward, done, _ = env.step_from_state(state, action)
        total_return += reward
        state = next_state

        # render the humanoid window
        env.env.render()
        time.sleep(sleep)

        print(
            f"t={t:03d} | reward={reward: .4f} | total={total_return: .4f} "
            f"| z={feat['z']: .4f} | vx={feat['vx']: .4f} "
            f"| upright={upright: .4f} | ||u||={np.linalg.norm(action): .4f}"
        )

        if done:
            print(f"\nEpisode terminated early at step {t}.")
            break

    print("\nFinal summary")
    print(f"Total return: {total_return:.6f}")
    print(f"Mean height z: {np.mean(heights):.6f}")
    print(f"Mean forward velocity vx: {np.mean(vxs):.6f}")
    print(f"Mean action norm: {np.mean(action_norms):.6f}")


def main():
    cfg = GPSConfig()
    cfg.render_mode = "human"

    env = HumanoidPaperEnv(cfg)

    policy = GaussianPolicy(
        obs_dim=env.obs_dim,
        act_dim=env.act_dim,
        cfg=cfg.policy,
    )

    checkpoint_path = "best_policy.pt"
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    policy.load_state_dict(state_dict)
    policy.eval()

    rollout_render(
        env=env,
        policy=policy,
        horizon=cfg.ilqr.horizon,
        deterministic=True,
        sleep=0.03,
    )


if __name__ == "__main__":
    main()