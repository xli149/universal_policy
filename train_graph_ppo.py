import os
import glob
import random
import torch  
from typing import Callable  
import gymnasium as gym
import gymnasium_env
from gymnasium.wrappers import TimeLimit
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

from graph_obs_wrapper import PaddedGraphObsWrapper, PaddedActionWrapper
from masked_graph_policy import MaskedGraphSACPolicy

print("🚀 正在启动达尔文计划：基础大脑预训练 (冷启动)")

# ==========================================
# 1. 📂 严格过滤：只放入 1、2、3 关节的“标准形态”做预训练
# ==========================================
pool_dir = "./gymnasium_env/envs/universal_pool"
xml_files = glob.glob(os.path.join(pool_dir, "*.xml"))

if len(xml_files) == 0:
    raise FileNotFoundError(f"🚨 在 {pool_dir} 找不到任何 XML 文件！")

# 🚨 课程学习阶段 1：只允许标准体型进入训练池！
ALLOWED_STANDARD_ENVS = [
    "reacher_1j_0.10.xml",
    "reacher_2j_0.10_0.10.xml",
    "reacher_3j_0.10_0.10_0.15.xml"
]

ALL_ENV_CONFIGS = []
for xml_path in xml_files:
    basename = os.path.basename(xml_path)
    if basename in ALLOWED_STANDARD_ENVS:
        joints_str = basename.split('_')[1]  
        n_joints = int(joints_str.replace('j', ''))
        ALL_ENV_CONFIGS.append({"xml": xml_path, "joints": n_joints})

random.seed(42)
random.shuffle(ALL_ENV_CONFIGS)

print(f"🌍 学前班构建完毕，共加载 {len(ALL_ENV_CONFIGS)} 个基础标准环境！")

TRAIN_CONFIGS = ALL_ENV_CONFIGS 
EVAL_CONFIGS = ALL_ENV_CONFIGS 

# ==========================================
# ⚙️ 基础配置区
# ==========================================
env_name = "gymnasium_env/Reacher2D-v0"
max_episode_steps = 100
total_timesteps = int(1e6)  # 预训练 100 万步足够了
seed = 0

scenario_name = "pretrained_base_brain_v1"
tb_log_dir = f"./tb_logs/{scenario_name}"
ckpt_dir = f"./checkpoints/{scenario_name}"

device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔥 计算设备: {device.upper()}\n")

def make_env(xml_file, n_arm_joints, render_mode=None):
    def _init():
        env = gym.make(env_name, xml_file=xml_file, render_mode=render_mode, frame_skip=4)
        env = TimeLimit(env, max_episode_steps=max_episode_steps)
        env = Monitor(env)
        env = PaddedGraphObsWrapper(env, max_joints=10, n_arm_joints=n_arm_joints) 
        env = PaddedActionWrapper(env, max_joints=10, n_arm_joints=n_arm_joints)
        return env
    return _init

train_env_fns = [make_env(cfg["xml"], cfg["joints"]) for cfg in TRAIN_CONFIGS]
eval_env_fns = [make_env(cfg["xml"], cfg["joints"]) for cfg in EVAL_CONFIGS]

train_env = VecMonitor(DummyVecEnv(train_env_fns))
eval_env = VecMonitor(DummyVecEnv(eval_env_fns))

n_envs = len(train_env_fns)

eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=ckpt_dir,     
    log_path=ckpt_dir,                 
    eval_freq=10_000, 
    n_eval_episodes=5, 
    deterministic=True,
    render=False
)

# ==========================================
# 🚨 终极拯救：锁死学习率，锁死探索率！
# ==========================================
model = SAC(
    policy=MaskedGraphSACPolicy,
    env=train_env,
    learning_rate=3e-4,     # ✅ 彻底放弃 linear_schedule，防止变僵尸！
    buffer_size=1_000_000, 
    batch_size=512,        
    ent_coef=0.05,          # ✅ 放弃 auto，设定强劲的 5% 探索率！
    gamma=0.99,
    tau=0.005,
    train_freq=(64, "step"), # 收集 64 步
    gradient_steps=16,       # 更新 16 次
    tensorboard_log=tb_log_dir,        
    verbose=1,
    seed=seed,
    device=device,
)

print(f"\n🚀 开始培育达尔文基础大脑 ...")
model.learn(total_timesteps=total_timesteps, callback=eval_callback, tb_log_name="run")

model.save(f"{ckpt_dir}/pretrained_base_brain")
train_env.close()
eval_env.close()
print("🎉 预训练结束！请将此模型用于后续的 GA-RL 协同进化！")