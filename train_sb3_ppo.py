import os
import gymnasium as gym
import gymnasium_env
from gymnasium.wrappers import TimeLimit

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

# ENV_ID = "gymnasium_env/Reacher2D-v5"
XML_FILE = "./gymnasium_env/envs/reacher_2j.xml"

env_name =  "gymnasium_env/Reacher2D-v5"
max_episode_steps = 100
total_timesteps = int(1e6)
seed = 0

def make_env(render_mode=None):
    def _init():
        env = gym.make(env_name, xml_file = XML_FILE, render_mode=render_mode)
        env = TimeLimit(env, max_episode_steps=max_episode_steps)
        env = Monitor(env)  # 记录 episode_return / episode_len 到 info 里
        return env
    return _init

# 训练环境（不渲染）
train_env = DummyVecEnv([make_env(render_mode=None)])
train_env = VecMonitor(train_env)  # VecEnv版本的monitor，tensorboard更稳

# 评估环境（可选：渲染 human 看效果；不渲染更快）
eval_env = DummyVecEnv([make_env(render_mode=None)])
eval_env = VecMonitor(eval_env)

# TensorBoard logdir
tb_log = os.path.join("sb3_runs", env_name, f"seed{seed}")

# callbacks：定期评估 + 定期保存
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=os.path.join("sb3_checkpoints", env_name, "best"),
    log_path=os.path.join("sb3_eval_logs", env_name),
    eval_freq=10_000,          # 每1万步评估一次（你可调大/调小）
    n_eval_episodes=5,
    deterministic=True,
    render=False
)

ckpt_callback = CheckpointCallback(
    save_freq=100_000,
    save_path=os.path.join("sb3_checkpoints", env_name, "ckpt"),
    name_prefix="ppo"
)

# SB3 PPO：先用默认 MlpPolicy
model = PPO(
    policy="MlpPolicy",
    env=train_env,
    learning_rate=3e-4,
    n_steps=2048,          # rollout长度（默认2048）
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.0,
    vf_coef=0.5,
    max_grad_norm=0.5,
    tensorboard_log=tb_log,
    verbose=1,
    seed=seed,
)

model.learn(
    total_timesteps=total_timesteps,
    callback=[eval_callback, ckpt_callback],
    tb_log_name="PPO"
)

model.save(os.path.join("sb3_checkpoints", env_name, "final_model"))

train_env.close()
eval_env.close()

print("Done. TensorBoard logdir:", tb_log)
