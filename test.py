import gymnasium as gym
env = gym.make("Humanoid-v4")
obs, info = env.reset()
print(obs.shape)