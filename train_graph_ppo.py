import os
import torch  
from typing import Callable  
import gymnasium as gym
import gymnasium_env
from gymnasium.wrappers import TimeLimit
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

# 导入你自定义的 Wrapper 和 Policy
from graph_obs_wrapper import PaddedGraphObsWrapper, PaddedActionWrapper
from masked_graph_policy import MaskedGraphSACPolicy

print("testing train graph ppo.py")

# ==========================================
# ⚙️ 核心配置区 (切换场景时，只需修改这两行！)
# ==========================================
XML_FILE = "./gymnasium_env/envs/reacher_3j.xml"  
N_ARM_JOINTS = 3  # 🚀 务必与 XML 文件里的真实手臂关节数保持一致！

env_name = "gymnasium_env/Reacher2D-v0"
max_episode_steps = 100
total_timesteps = int(1e6)
seed = 0

# 🚀 自动提取场景名称 (比如提取出 "reacher_2j")
scenario_name = os.path.splitext(os.path.basename(XML_FILE))[0]

# 📁 极简的目录路径设计
tb_log_dir = f"./tb_logs/{scenario_name}"
ckpt_dir = f"./checkpoints/{scenario_name}"

print(f"\n📁 当前实验场景: {scenario_name}, 真实关节数: {N_ARM_JOINTS}")

# ==========================================
# 自动检测计算设备 (MPS/CUDA/CPU)
# ==========================================
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
print(f"🔥 准备使用的计算设备: {device.upper()}\n")

# 定义线性学习率衰减
def linear_schedule(initial_value: float) -> Callable[[float], float]:
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func

def make_env(render_mode=None):
    def _init():
        env = gym.make(env_name, xml_file=XML_FILE, render_mode=render_mode)
        env = TimeLimit(env, max_episode_steps=max_episode_steps)
        env = Monitor(env)
        
        # 🚀 动态传入真实关节数，拒绝硬编码
        env = PaddedGraphObsWrapper(env, max_joints=10, n_arm_joints=N_ARM_JOINTS) 
        env = PaddedActionWrapper(env, max_joints=10, n_arm_joints=N_ARM_JOINTS)
        return env
    return _init

train_env = VecMonitor(DummyVecEnv([make_env(render_mode=None)]))
eval_env = VecMonitor(DummyVecEnv([make_env(render_mode=None)]))

# ==========================================
# 设置回调函数：只保留最有用的最高分模型存档
# ==========================================
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=ckpt_dir,     # 🚀 最好的模型直接存到对应的场景文件夹里
    log_path=ckpt_dir,                 # 评估的 numpy 成绩单也存在这里
    eval_freq=10_000,
    n_eval_episodes=5,
    deterministic=True,
    render=False
)

# 使用 SAC 训练策略
model = SAC(
    policy=MaskedGraphSACPolicy,
    env=train_env,
    learning_rate=linear_schedule(3e-4),
    buffer_size=100_000,
    batch_size=256,        
    ent_coef=0.01,       
    gamma=0.99,
    tau=0.005,
    tensorboard_log=tb_log_dir,        # 🚀 指向极简的日志文件夹
    verbose=1,
    seed=seed,
    device=device,
)

print(f"开始使用 GCN SAC 训练 {scenario_name} ...")
model.learn(
    total_timesteps=total_timesteps,
    callback=eval_callback,
    tb_log_name="run"                  # TensorBoard 里会显示 run_1, run_2
)

# ==========================================
# 保存收敛瞬间的最终模型
# ==========================================
model.save(f"{ckpt_dir}/final_model")
train_env.close()
eval_env.close()
print(f"训练完成。TensorBoard logdir: {tb_log_dir}")