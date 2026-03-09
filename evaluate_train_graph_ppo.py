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

# 🚀 目标锁定：指向刚刚炼成的“学前班基础大脑”
MODEL_PATH = "./checkpoints/pretrained_base_brain_v1/pretrained_base_brain" 

# 🚀 评估测试池：严格对齐训练时的 1、2、3 关节标准形态
POOL_DIR = "./gymnasium_env/envs/universal_pool"
ENV_CONFIGS = [
    {"xml": os.path.join(POOL_DIR, "reacher_1j_0.10.xml"), "joints": 1},
    {"xml": os.path.join(POOL_DIR, "reacher_2j_0.10_0.10.xml"), "joints": 2},
   {"xml": os.path.join(POOL_DIR, "reacher_4j_0.05_0.05_0.15_0.10.xml"), "joints": 4},
]

max_episode_steps = 100 
max_joints = 10

# ==========================================
# 动态生成环境的工厂函数
# ==========================================
def make_eval_env(xml_file, n_arm_joints, render_mode="human"):
    def _init():
        # ✅ 核心修复：必须加上 frame_skip=4，保持物理世界时钟一致！
        env = gym.make(ENV_ID, xml_file=xml_file, render_mode=render_mode, frame_skip=4)
        env = TimeLimit(env, max_episode_steps=max_episode_steps)
        env = PaddedGraphObsWrapper(env, max_joints=max_joints, n_arm_joints=n_arm_joints)
        env = PaddedActionWrapper(env, max_joints=max_joints, n_arm_joints=n_arm_joints)
        return env
    return _init

# ==========================================
# 单个环境的评估循环
# ==========================================
def eval_on_xml(model, xml_file, n_arm_joints, n_episodes=5):
    venv = DummyVecEnv([make_eval_env(xml_file, n_arm_joints, render_mode="human")])
    raw_env = venv.envs[0].unwrapped
    env_success_th = float(getattr(raw_env, "success_threshold", 0.10)) # 统一判断阈值

    ep_rews, final_dists = [], []
    success_count = 0

    for ep in range(n_episodes):
        obs = venv.reset() 
        done = False
        ep_rew = 0.0
        final_ep_dist = 999.0  # 🌟 用来暂存真正的回合末尾距离

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done_arr, info = venv.step(action)
            ep_rew += float(reward[0])
            done = done_arr[0]
            
            # 🌟 修复自动重置陷阱：趁着 DummyVecEnv 重置前，把真实的最终距离截获下来！
            if "dist" in info[0]:
                final_ep_dist = float(info[0]["dist"])

        ep_rews.append(ep_rew)
        final_dists.append(final_ep_dist)
        
        is_success = final_ep_dist < env_success_th
        if is_success:
            success_count += 1

        print(f"Episode {ep+1:02d}: Reward={ep_rew:.2f}, Final Dist={final_ep_dist:.4f}, Success={is_success}")

    venv.close()

    return {
        "xml": os.path.basename(xml_file),
        "joints": n_arm_joints,
        "ep_rew_mean": float(np.mean(ep_rews)),
        "final_dist_mean": float(np.mean(final_dists)),
        "success_rate": float(success_count / n_episodes),
    }

# ==========================================
# 主程序：验收基础大脑
# ==========================================
if __name__ == "__main__":
    if not os.path.exists(MODEL_PATH + ".zip"):
        print(f"🚨 致命错误: 找不到模型文件 {MODEL_PATH}.zip")
        exit(1)
        
    # 为了让 SB3 正确初始化，先建一个临时空壳环境（不渲染）
    first_cfg = ENV_CONFIGS[0]
    temp_env = DummyVecEnv([make_eval_env(first_cfg["xml"], first_cfg["joints"], render_mode=None)])
    
    print(f"Loading Universal Model from {MODEL_PATH}...")
    model = SAC.load(MODEL_PATH, env=temp_env, device="auto")
    temp_env.close()

    print("\n=== 验收学前班基础大脑 ===")
    
    for config in ENV_CONFIGS:
        print(f"\n🎬 正在测试环境: {os.path.basename(config['xml'])} (关节数: {config['joints']})")
        metrics = eval_on_xml(model, xml_file=config["xml"], n_arm_joints=config["joints"], n_episodes=5)
        
        print("\n" + "="*40)
        print(f"🏆 {config['joints']} 关节成绩单:")
        for k, v in metrics.items():
            print(f"  - {k}: {v:.4f}" if isinstance(v, float) else f"  - {k}: {v}")
        print("="*40)