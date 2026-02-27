import os
import numpy as np
import gymnasium as gym
import gymnasium_env
from gymnasium.wrappers import TimeLimit

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv

# 导入自定义的 Wrapper 和 Policy
from graph_obs_wrapper import PaddedGraphObsWrapper, PaddedActionWrapper
from masked_graph_policy import MaskedGraphSACPolicy

# ==========================================
# ⚙️ 核心配置区
# ==========================================
ENV_ID = "gymnasium_env/Reacher2D-v0"

# 🚀 指向你刚刚跑完的“混合训练大模型”路径
MODEL_PATH = "./checkpoints/universal_2_to_5j/final_model" # 或者 best_model

# 🚀 评估测试池：让模型依次挑战 2 到 5 关节
ENV_CONFIGS = [
    {"xml": "./gymnasium_env/envs/reacher_2j.xml", "joints": 2},
    {"xml": "./gymnasium_env/envs/reacher_3j.xml", "joints": 3},
    {"xml": "./gymnasium_env/envs/reacher_4j.xml", "joints": 4},
    {"xml": "./gymnasium_env/envs/reacher_5j.xml", "joints": 5},
]

max_episode_steps = 100 
max_joints = 10

# ==========================================
# 动态生成环境的工厂函数
# ==========================================
# 🚀 修改 1：接收动态的 n_arm_joints，每次创建不同长短的手臂
def make_eval_env(xml_file, n_arm_joints, render_mode="human"):
    def _init():
        env = gym.make(ENV_ID, xml_file=xml_file, render_mode=render_mode)
        env = TimeLimit(env, max_episode_steps=max_episode_steps)
        
        # 精确贴合当前环境的关节数
        env = PaddedGraphObsWrapper(env, max_joints=max_joints, n_arm_joints=n_arm_joints)
        env = PaddedActionWrapper(env, max_joints=max_joints, n_arm_joints=n_arm_joints)
        
        return env
    return _init

# ==========================================
# 单个环境的评估循环
# ==========================================
def eval_on_xml(model, xml_file, n_arm_joints, n_episodes=5):
    # 用当前的 config 生成特定的环境
    venv = DummyVecEnv([make_eval_env(xml_file, n_arm_joints, render_mode="human")])
    raw_env = venv.envs[0].unwrapped
    env_success_th = float(getattr(raw_env, "success_threshold", 0.05))

    ep_rews, final_dists = [], []
    success_count = 0

    for ep in range(n_episodes):
        obs = venv.reset() 
        done = False
        ep_rew = 0.0

        while not done:
            # 🚀 GCN 开始表演：deterministic=True 直接输出当前最优策略
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done_arr, info = venv.step(action)
            ep_rew += float(reward[0])
            done = done_arr[0]

        # 计算最后停靠时的距离误差
        if hasattr(raw_env, "_get_dist"):
            dist = float(raw_env._get_dist())
        else:
            fingertip = raw_env.get_body_com("fingertip")
            target = raw_env.get_body_com("target")
            dist = float(np.linalg.norm(fingertip - target))

        ep_rews.append(ep_rew)
        final_dists.append(dist)
        
        is_success = dist < env_success_th
        if is_success:
            success_count += 1

        print(f"Episode {ep+1:02d}: Reward={ep_rew:.2f}, Final Dist={dist:.4f}, Success={is_success}")

    venv.close()

    return {
        "xml": os.path.basename(xml_file),
        "joints": n_arm_joints,
        "ep_rew_mean": float(np.mean(ep_rews)),
        "final_dist_mean": float(np.mean(final_dists)),
        "success_rate": float(success_count / n_episodes),
    }

# ==========================================
# 主程序：遍历全宇宙
# ==========================================
if __name__ == "__main__":
    # 为了让 SB3 正确初始化，先用第一个配置建一个临时空壳环境（不渲染）
    first_cfg = ENV_CONFIGS[0]
    temp_env = DummyVecEnv([make_eval_env(first_cfg["xml"], first_cfg["joints"], render_mode=None)])
    
    print(f"Loading Universal Model from {MODEL_PATH}...")
    model = SAC.load(MODEL_PATH, env=temp_env, device="auto")
    temp_env.close()

    print("\n=== Start Universal Evaluation ===")
    
    # 🚀 修改 2：依次遍历 2、3、4、5 关节的环境
    for config in ENV_CONFIGS:
        print(f"\n🎬 正在测试环境: {os.path.basename(config['xml'])} (关节数: {config['joints']})")
        
        # 每个环境跑 5 局看看效果
        metrics = eval_on_xml(model, xml_file=config["xml"], n_arm_joints=config["joints"], n_episodes=5)
        
        print("\n" + "="*40)
        print(f"🏆 {config['joints']} 关节成绩单:")
        for k, v in metrics.items():
            print(f"  - {k}: {v:.4f}" if isinstance(v, float) else f"  - {k}: {v}")
        print("="*40)