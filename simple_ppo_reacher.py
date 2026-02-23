import gymnasium as gym
from stable_baselines3 import PPO

env = gym.make("Reacher-v5")

model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    n_steps=2048,
    batch_size=64,
    gae_lambda=0.95,
    gamma=0.99,
    n_epochs=10,
    ent_coef=0.0,
    learning_rate=3e-4,
    clip_range=0.2,
)

model.learn(total_timesteps=1_000_000)
model.save("ppo_reacher_v5")

env.close()
