import gymnasium as gym
import gymnasium_env
from gymnasium.wrappers import TimeLimit
from stable_baselines3 import PPO

env_name = "Reacher-v5"
max_episode_steps = 1000

# 1) 创建带 human 渲染的环境
env = gym.make(env_name, render_mode="human")
env = TimeLimit(env, max_episode_steps=max_episode_steps)

# 2) 加载你训练好的模型（把路径改成你的实际文件）
model_path = f"sb3_checkpoints/{env_name}/final_model"
model = PPO.load(model_path)

# 3) 跑几条 episode
num_episodes = 5

for ep in range(num_episodes):
    obs, info = env.reset()
    done = False
    ep_reward = 0.0
    steps = 0

    while not done:
        # deterministic=True：用确定性策略看效果更稳定
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        ep_reward += float(reward)
        steps += 1

    print(f"[EP {ep}] reward={ep_reward:.2f}, steps={steps}")

env.close()
