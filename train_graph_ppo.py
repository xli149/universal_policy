import gymnasium as gym
import gymnasium_env
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from graph_obs_wrapper import PaddedGraphObsWrapper
from masked_graph_policy import MaskedGraphSACPolicy # ✅ 引入新的 SAC Policy

def make_env(xml_file):
    def _init():
        env = gym.make("gymnasium_env/Reacher2D-v0", xml_file=xml_file)
        from gymnasium.wrappers import TimeLimit
        env = TimeLimit(env, max_episode_steps=50)
        env = PaddedGraphObsWrapper(env, max_joints=10)
        return env
    return _init

def train():
    XML_2J = "/Users/chrislee/Documents/mujoco_test/gymnasium_env/envs/reacher_2j.xml"
    
    # SAC 通常对环境并行的依赖没有 PPO 那么重，开 4-8 个均可
    venv = DummyVecEnv([make_env(XML_2J) for _ in range(8)])
    venv = VecMonitor(venv)

    model = SAC(
        policy=MaskedGraphSACPolicy,
        env=venv,
        learning_rate=3e-4,
        buffer_size=100000,     # ✅ SAC 灵魂：经验回放池
        batch_size=256,
        ent_coef=0.02,        # ✅ SAC 魔法：让它自己调探索欲望！
        gamma=0.99,
        tau=0.005,              # 目标网络软更新
        train_freq=1,           # 每走 1 步就拿回放池的数据训练 1 次
        gradient_steps=1,
        verbose=1,
        tensorboard_log="./sac_reacher_tensorboard/"
    )

    print("🚀 Starting SAC Training...")
    model.learn(total_timesteps=500000, log_interval=4)
    model.save("./checkpoints/graph_reach_sac_final")
    print(f"Model saved to ./checkpoints/graph_reach_sac_final ")

if __name__ == "__main__":
    train()