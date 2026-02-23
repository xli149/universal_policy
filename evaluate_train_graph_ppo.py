# eval_universal_policy_all_human.py
import os
import numpy as np
import gymnasium as gym
import gymnasium_env
from gymnasium.wrappers import TimeLimit

# ✅ 修改 1：将 PPO 换成 SAC
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv

from graph_obs_wrapper import PaddedGraphObsWrapper

# ✅ 修改 2：引入我们为 SAC 新写的 Policy 类 (虽然评估时其实由 SB3 自动反序列化，但保持引入是个好习惯)
from masked_graph_policy import MaskedGraphSACPolicy  

ENV_ID = "gymnasium_env/Reacher2D-v0"

# ✅ 修改 3：指向你刚刚训练好的 SAC 最新模型文件
MODEL_PATH = "./checkpoints/graph_reach_sac_final" 

XML_POOL = [
    "/Users/chrislee/Documents/mujoco_test/gymnasium_env/envs/reacher_2j.xml",
    # 测试多具身时，可以随时把下面两行取消注释
    # "/Users/chrislee/Documents/mujoco_test/gymnasium_env/envs/reacher_3j.xml",
    # "/Users/chrislee/Documents/mujoco_test/gymnasium_env/envs/reacher_4j.xml",
]

max_episode_steps = 50
max_joints = 10

def make_eval_env(xml_file, render_mode="human"):
    def _init():
        env = gym.make(ENV_ID, xml_file=xml_file, render_mode=render_mode)
        env = TimeLimit(env, max_episode_steps=max_episode_steps)
        # Wrapper 会自动处理 5 维特征和 10 维动作的对齐
        env = PaddedGraphObsWrapper(env, max_joints=max_joints)
        return env
    return _init

def eval_on_xml(model, xml_file, n_episodes=20):
    """
    执行评估：直接使用 SB3 原生 predict API
    """
    venv = DummyVecEnv([make_eval_env(xml_file, render_mode="human")])
    raw_env = venv.envs[0].unwrapped
    env_success_th = float(getattr(raw_env, "success_threshold", 0.1))

    ep_rews, final_dists = [], []
    success_count = 0

    for ep in range(n_episodes):
        obs = venv.reset() 
        done = False
        ep_rew = 0.0

        while not done:
            # ✅ 对于 SAC，deterministic=True 意味着它不再进行随机探索，而是输出最确信的均值 (Mean) 动作
            action, _ = model.predict(obs, deterministic=True)
            
            # 直接将 action 喂给 venv 即可，无需加 [ ]
            obs, reward, done_arr, info = venv.step(action)
            
            ep_rew += float(reward[0])
            done = done_arr[0]

        # 获取最终距离
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
    # 1. 创建临时空环境用于加载模型结构
    temp_env = DummyVecEnv([make_eval_env(XML_POOL[0], render_mode=None)])
    
    print(f"Loading model from {MODEL_PATH}...")
    
    # ✅ 修改 4：使用 SAC.load 来加载你的新模型
    model = SAC.load(MODEL_PATH, env=temp_env, device="auto")
    temp_env.close()

    print("\n=== Start Evaluation ===")
    for xml in XML_POOL:
        print(f"\nTesting on: {os.path.basename(xml)}")
        metrics = eval_on_xml(model, xml_file=xml, n_episodes=20)
        
        print("\n" + "="*30)
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")
        print("="*30)