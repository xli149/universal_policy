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
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, SubprocVecEnv

# 导入你自定义的 Wrapper 和 Policy
from graph_obs_wrapper import PaddedGraphObsWrapper, PaddedActionWrapper
from masked_graph_policy import MaskedGraphSACPolicy

print("🚀 正在启动达尔文计划：1-10关节通用大模型训练")

# ==========================================
# 1. 📂 动态扫描大千世界环境池
# ==========================================
pool_dir = "./gymnasium_env/envs/universal_pool"
xml_files = glob.glob(os.path.join(pool_dir, "*.xml"))

if len(xml_files) == 0:
    raise FileNotFoundError(f"🚨 在 {pool_dir} 找不到任何 XML 文件，请先运行生成脚本！")

ALL_ENV_CONFIGS = []
for xml_path in xml_files:
    # 从文件名解析关节数量 (例如: reacher_4j_0.10_0.10.xml -> 提取出 4)
    basename = os.path.basename(xml_path)
    joints_str = basename.split('_')[1]  # 提取 "4j"
    n_joints = int(joints_str.replace('j', ''))
    ALL_ENV_CONFIGS.append({"xml": xml_path, "joints": n_joints})

# 随机打乱环境顺序，防止网络产生强烈的先后记忆偏见
random.seed(42)
random.shuffle(ALL_ENV_CONFIGS)

print(f"🌍 成功扫描到 {len(ALL_ENV_CONFIGS)} 个形态各异的机械臂环境！")

# ==========================================
# 2. ⚖️ 拆分训练集与评估集 (防止评估耗时过长)
# ==========================================
# 为了评估快一点，我们从 1-10 关节中，每种关节数只挑 1 个最具代表性的环境组成“评估集”(共10个)
EVAL_CONFIGS = []
for j in range(1, 11):
    for cfg in ALL_ENV_CONFIGS:
        if cfg["joints"] == j:
            EVAL_CONFIGS.append(cfg)
            break

# 剩下的全部扔进“训练集” (也可以全用，这里全部放入训练集)
TRAIN_CONFIGS = ALL_ENV_CONFIGS 

# ==========================================
# ⚙️ 基础配置区
# ==========================================
env_name = "gymnasium_env/Reacher2D-v0"
max_episode_steps = 100
# 🚀 1到10关节难度极其变态，建议 500 万步起步！
total_timesteps = int(5e6) 
seed = 0

scenario_name = "universal_1_to_10j_v1"
tb_log_dir = f"./tb_logs/{scenario_name}"
ckpt_dir = f"./checkpoints/{scenario_name}"

print(f"📁 当前实验场景: {scenario_name}")

# 自动检测计算设备
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
print(f"🔥 计算设备: {device.upper()}\n")

def linear_schedule(initial_value: float) -> Callable[[float], float]:
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func

# ==========================================
# 3. 🏭 动态环境工厂
# ==========================================
def make_env(xml_file, n_arm_joints, render_mode=None):
    def _init():
        # 这里保留了我们之前加的 frame_skip=4
        env = gym.make(env_name, xml_file=xml_file, render_mode=render_mode, frame_skip=4)
        env = TimeLimit(env, max_episode_steps=max_episode_steps)
        env = Monitor(env)
        env = PaddedGraphObsWrapper(env, max_joints=10, n_arm_joints=n_arm_joints) 
        env = PaddedActionWrapper(env, max_joints=10, n_arm_joints=n_arm_joints)
        return env
    return _init

# ==========================================
# 4. 🚀 构建并行 VecEnv
# ==========================================
# 生成函数列表
train_env_fns = [make_env(cfg["xml"], cfg["joints"]) for cfg in TRAIN_CONFIGS]
eval_env_fns = [make_env(cfg["xml"], cfg["joints"]) for cfg in EVAL_CONFIGS]

# 使用 DummyVecEnv 装载大千世界 (MuJoCo 非常快，200 个用 Dummy 也跑得飞起)
train_env = VecMonitor(DummyVecEnv(train_env_fns))
eval_env = VecMonitor(DummyVecEnv(eval_env_fns))

n_envs = len(train_env_fns)
print(f"⚔️ 训练环境并发数: {n_envs}")
print(f"🎯 评估环境并发数: {len(eval_env_fns)}")

# ==========================================
# 5. 设置回调与模型
# ==========================================
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=ckpt_dir,     
    log_path=ckpt_dir,                 
    eval_freq=max(10_000 // n_envs, 1), # 动态调整评估频率
    n_eval_episodes=5, 
    deterministic=True,
    render=False
)

model = SAC(
    policy=MaskedGraphSACPolicy,
    env=train_env,
    learning_rate=linear_schedule(3e-4),
    buffer_size=1_000_000, # 🚀 缓冲池加大到 100 万，容纳大千世界的数据
    batch_size=512,        # 🚀 批次加大，保证图网络有足够的梯度
    ent_coef=0.01,       
    gamma=0.99,
    tau=0.005,
    # ==========================================================
    # 🚨 极其关键的修改：防止梯度饥饿！
    # 因为有 n_envs 个并行环境，每次 step 会收集 n_envs 条数据。
    # 我们必须把 gradient_steps 设为 n_envs，保证收集多少步，就训练多少次！
    # ==========================================================
    train_freq=(1, "step"),
    gradient_steps=n_envs, 
    tensorboard_log=tb_log_dir,        
    verbose=1,
    seed=seed,
    device=device,
)

print(f"\n🚀 开始使用 GCN SAC 征服大千世界 ...")
model.learn(
    total_timesteps=total_timesteps,
    callback=eval_callback,
    tb_log_name="run"                  
)

model.save(f"{ckpt_dir}/final_model")
train_env.close()
eval_env.close()
print(f"🎉 训练完美收官！TensorBoard logdir: {tb_log_dir}")