from __future__ import annotations

import time
from pathlib import Path

import gymnasium as gym
from stable_baselines3 import PPO


ENV_ID = "Humanoid-v5"   # change to v4 if needed
MODEL_PATH = Path("models/best/best_model.zip")  # or models/ppo_humanoid_final.zip


def main():
    env = gym.make(ENV_ID, render_mode="human")
    model = PPO.load(str(MODEL_PATH))

    obs, info = env.reset()
    total_reward = 0.0

    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Small sleep makes the render easier to watch
        time.sleep(0.01)

        if terminated or truncated:
            print(f"Episode return: {total_reward:.3f}")
            obs, info = env.reset()
            total_reward = 0.0


if __name__ == "__main__":
    main()