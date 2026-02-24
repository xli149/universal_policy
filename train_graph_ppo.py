import os
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
XML_FILE = "./gymnasium_env/envs/reacher_2j.xml"  # 使用你修改好物理参数的 XML
env_name = "gymnasium_env/Reacher2D-v0"
max_episode_steps = 100
total_timesteps = int(1e6)
seed = 0

def make_env(render_mode=None):
    def _init():
        # 注意：这里需要你确保底层 _get_obs 返回的是一个完整的字典，
        # 或者直接让你的 Wrapper 处理原始的 dict。
        env = gym.make(env_name, xml_file=XML_FILE, render_mode=render_mode)
        env = TimeLimit(env, max_episode_steps=max_episode_steps)
        env = Monitor(env)
        env = PaddedGraphObsWrapper(env, max_joints=10) # 包装环境！
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

# 使用 SAC 训练 Transformer 策略
model = SAC(
    policy=MaskedGraphSACPolicy,
    # policy = "MultiInputPolicy",
    env=train_env,
    learning_rate=3e-4,
    buffer_size=100_000,   # SAC 经验回放池
    batch_size=256,        # 🚀 增大 Batch Size 以稳定 Transformer 的梯度
    ent_coef="auto",       # 自动调节熵，鼓励探索
    # target_entropy=-2.0,
    gamma=0.99,
    tau=0.005,
    tensorboard_log=tb_log,
    verbose=1,
    seed=seed,
)

print("开始使用 Transformer SAC 训练...")
model.learn(
    total_timesteps=total_timesteps,
    callback=[eval_callback, ckpt_callback],
    tb_log_name="SAC_GNN"
)

model.save(os.path.join("sb3_checkpoints", env_name, "final_sac_gnn_model"))
train_env.close()
eval_env.close()
print("训练完成。TensorBoard logdir:", tb_log)