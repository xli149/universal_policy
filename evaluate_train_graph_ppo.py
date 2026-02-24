import os
import numpy as np
import gymnasium as gym
import gymnasium_env
from gymnasium.wrappers import TimeLimit

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv

# ✅ 修改 1：同时导入 Observation 和 Action 的 Wrapper
from graph_obs_wrapper import PaddedGraphObsWrapper, PaddedActionWrapper

# ✅ 修改 2：必须导入你自定义的 Policy，否则 SB3 加载模型时会找不到类！
from masked_graph_policy import MaskedGraphSACPolicy

# 确保这里的 ENV_ID 和你训练时注册的名字一致（比如 v5）
ENV_ID = "gymnasium_env/Reacher2D-v0"

# ✅ 修改 3：指向你刚刚跑完的 GCN 模型路径 (不需要加 .zip 后缀)
MODEL_PATH = "./sb3_checkpoints/gymnasium_env/Reacher2D-v0/final_sac_gnn_model" 

XML_POOL = [
    "/Users/chrislee/Documents/mujoco_test/gymnasium_env/envs/reacher_2j.xml",
]

max_episode_steps = 100 # 保持和训练时的 max_episode_steps 一致
max_joints = 10
n_arm_joints = 2        # 明确真实的机械臂关节数

def make_eval_env(xml_file, render_mode="human"):
    def _init():
        env = gym.make(ENV_ID, xml_file=xml_file, render_mode=render_mode)
        env = TimeLimit(env, max_episode_steps=max_episode_steps)
        
        # ✅ 修改 4：必须双管齐下，既包装观察，又包装动作！并传入 n_arm_joints
        env = PaddedGraphObsWrapper(env, max_joints=max_joints, n_arm_joints=n_arm_joints)
        env = PaddedActionWrapper(env, max_joints=max_joints, n_arm_joints=n_arm_joints)
        
        return env
    return _init

def eval_on_xml(model, xml_file, n_episodes=10):
    venv = DummyVecEnv([make_eval_env(xml_file, render_mode="human")])
    raw_env = venv.envs[0].unwrapped
    env_success_th = float(getattr(raw_env, "success_threshold", 0.05)) # 容忍距离

    ep_rews, final_dists = [], []
    success_count = 0

    for ep in range(n_episodes):
        obs = venv.reset() 
        done = False
        ep_rew = 0.0

        while not done:
            # deterministic=True 代表评估时不加入探索噪音，直接输出最优动作
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done_arr, info = venv.step(action)
            ep_rew += float(reward[0])
            done = done_arr[0]

        # 获取最终的指尖到目标的距离
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
        "ep_rew_mean": float(np.mean(ep_rews)),
        "final_dist_mean": float(np.mean(final_dists)),
        "success_rate": float(success_count / n_episodes),
    }

if __name__ == "__main__":
    # 加载模型时不需要渲染画面
    temp_env = DummyVecEnv([make_eval_env(XML_POOL[0], render_mode=None)])
    
    print(f"Loading model from {MODEL_PATH}...")
    model = SAC.load(MODEL_PATH, env=temp_env, device="auto")
    temp_env.close()

    print("\n=== Start Evaluation ===")
    for xml in XML_POOL:
        print(f"\nTesting on: {os.path.basename(xml)}")
        # 弹窗渲染评估
        metrics = eval_on_xml(model, xml_file=xml, n_episodes=10)
        
        print("\n" + "="*30)
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")
        print("="*30)