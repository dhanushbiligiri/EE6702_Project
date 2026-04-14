from __future__ import annotations

import os
from pathlib import Path

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor


ENV_ID = "Humanoid-v4"   # change to "Humanoid-v4" if you specifically want v4
TOTAL_TIMESTEPS = 10_000_000
MODEL_DIR = Path("models")
LOG_DIR = Path("logs")


def make_env(render_mode=None):
    def _thunk():
        env = gym.make(ENV_ID, render_mode=render_mode)
        env = Monitor(env)
        return env
    return _thunk


def main():
    MODEL_DIR.mkdir(exist_ok=True)
    LOG_DIR.mkdir(exist_ok=True)

    train_env = DummyVecEnv([make_env()])
    train_env = VecMonitor(train_env)

    eval_env = DummyVecEnv([make_env()])
    eval_env = VecMonitor(eval_env)

    checkpoint_callback = CheckpointCallback(
        save_freq=50_000,
        save_path=str(MODEL_DIR / "checkpoints"),
        name_prefix="ppo_humanoid",
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(MODEL_DIR / "best"),
        log_path=str(LOG_DIR / "eval"),
        eval_freq=25_000,
        deterministic=True,
        render=False,
    )

    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        tensorboard_log=str(LOG_DIR / "tb"),
    )

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=[checkpoint_callback, eval_callback],
        progress_bar=True,
    )

    model.save(str(MODEL_DIR / "ppo_humanoid_final"))
    print(f"Saved final model to {MODEL_DIR / 'ppo_humanoid_final.zip'}")


if __name__ == "__main__":
    main()