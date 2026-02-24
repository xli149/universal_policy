import os
import gymnasium as gym
import gymnasium_env
import time
from stable_baselines3 import PPO
from gymnasium.wrappers import TimeLimit

# --- 配置 ---
env_name = "gymnasium_env/Reacher2D-v5"
XML_FILE = "./gymnasium_env/envs/reacher_2j.xml"
# 找到你训练时保存的模型路径
model_path = os.path.join("sb3_checkpoints", env_name, "best", "best_model.zip") 
# 或者使用最后的模型: os.path.join("sb3_checkpoints", env_name, "final_model.zip")

def evaluate():
    # 1. 创建用于可视化的环境
    # 注意：渲染模式设为 "human"，这样会自动弹出窗口
    env = gym.make(env_name, xml_file = XML_FILE, render_mode="human")
    env = TimeLimit(env,max_episode_steps = 100)

    # 如果你的训练脚本里用了 TimeLimit 包装器，这里最好保持一致
    # 但对于可视化，不加通常也能运行，直到环境自然结束
    
    # 2. 加载模型
    if not os.path.exists(model_path):
        print(f"Error: 找不到模型文件 {model_path}")
        return

    model = PPO.load(model_path)
    print(f"成功加载模型: {model_path}")

    # 3. 运行几个回合
    num_episodes = 5
    for i in range(num_episodes):
        obs, info = env.reset()
        done = False
        truncated = False
        episode_reward = 0
        
        print(f"开始第 {i+1} 个回合...")
        
        while not (done or truncated):
            # predict 会返回 action 和 states（后者用于 RNN，PPO 默认不需要）
            # deterministic=True 确保测试时动作是确定性的，表现更稳
            action, _states = model.predict(obs, deterministic=True)
            
            # 与环境交互
            obs, reward, done, truncated, info = env.step(action)
            print(f"done:{done}, truncated: {truncated}")
            episode_reward += reward
            
            # 控制一下渲染速度，否则 MuJoCo 可能跑得太快看不清
            time.sleep(0.02) 
            
        print(f"回合结束，总奖励: {episode_reward:.2f}")

    env.close()
    print("评估完成。")

if __name__ == "__main__":
    evaluate()