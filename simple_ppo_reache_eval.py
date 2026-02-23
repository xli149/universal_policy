import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

env = gym.make("Reacher-v5")  # 你的自定义 env 也可以换成对应 id

model = PPO.load("ppo_reacher_v5", env=env)

mean_reward, std_reward = evaluate_policy(
    model,
    env,
    n_eval_episodes=20,
    deterministic=True,   # 评估时通常用确定性动作（均值动作）
    render=False
)

print(f"mean_reward={mean_reward:.2f} +/- {std_reward:.2f}")
env.close()
