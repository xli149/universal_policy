import os
import torch  # 🚀 新增：用于检测设备
from typing import Callable  # 🚀 新增：用于定义学习率衰减
import gymnasium as gym
import gymnasium_env
from gymnasium.wrappers import TimeLimit
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

# 导入你自定义的 Wrapper 和 Policy
from graph_obs_wrapper import PaddedGraphObsWrapper, PaddedActionWrapper
from masked_graph_policy import MaskedGraphSACPolicy

print(f"testing train graph ppo.py")
XML_FILE = "./gymnasium_env/envs/reacher_2j.xml"  
env_name = "gymnasium_env/Reacher2D-v0"
max_episode_steps = 100
total_timesteps = int(1e6)
seed = 0

# ==========================================
# 🚀 进阶技巧 1：自动检测计算设备 (MPS/CUDA/CPU)
# ==========================================
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
print(f"\n🔥 准备使用的计算设备: {device.upper()}\n")

# ==========================================
# 🚀 进阶技巧 2：定义线性学习率衰减
# ==========================================
def linear_schedule(initial_value: float) -> Callable[[float], float]:
    def func(progress_remaining: float) -> float:
        # progress_remaining 从 1.0 线性降到 0.0
        return progress_remaining * initial_value
    return func

def make_env(render_mode=None):
    def _init():
        env = gym.make(env_name, xml_file=XML_FILE, render_mode=render_mode)
        env = TimeLimit(env, max_episode_steps=max_episode_steps)
        env = Monitor(env)
        
        # 🚀 修复隐患：PaddedGraphObsWrapper 之前漏掉了 n_arm_joints=2，必须加上！
        env = PaddedGraphObsWrapper(env, max_joints=10, n_arm_joints=2) 
        env = PaddedActionWrapper(env, max_joints=10, n_arm_joints=2)
        return env
    return _init

train_env = VecMonitor(DummyVecEnv([make_env(render_mode=None)]))
eval_env = VecMonitor(DummyVecEnv([make_env(render_mode=None)]))

tb_log = os.path.join("sb3_runs", env_name, f"sac_gnn_seed{seed}")

eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=os.path.join("sb3_checkpoints", env_name, "best_sac_gnn"),
    log_path=os.path.join("sb3_eval_logs", env_name),
    eval_freq=10_000,
    n_eval_episodes=5,
    deterministic=True,
    render=False
)

ckpt_callback = CheckpointCallback(
    save_freq=100_000,
    save_path=os.path.join("sb3_checkpoints", env_name, "ckpt"),
    name_prefix="sac_gnn"
)

# 使用 SAC 训练策略
model = SAC(
    policy=MaskedGraphSACPolicy,
    env=train_env,
    # 🚀 进阶技巧应用：使用学习率衰减，从 3e-4 平滑降至 0，便于后期逼近极限 -3 分
    learning_rate=linear_schedule(3e-4),
    buffer_size=100_000,
    batch_size=256,        
    
    # 🚀 进阶技巧 3：固定探索系数。舍弃 "auto"，防止熵跌到 0 导致模型摆烂
    ent_coef=0.01,       
    
    gamma=0.99,
    tau=0.005,
    tensorboard_log=tb_log,
    verbose=1,
    seed=seed,
    
    # 🚀 指定使用的硬件设备
    device=device,
)

print("开始使用 GCN SAC 训练...")
model.learn(
    total_timesteps=total_timesteps,
    callback=[eval_callback, ckpt_callback],
    tb_log_name="SAC_GNN"
)

model.save(os.path.join("sb3_checkpoints", env_name, "final_sac_gnn_model"))
train_env.close()
eval_env.close()
print("训练完成。TensorBoard logdir:", tb_log)