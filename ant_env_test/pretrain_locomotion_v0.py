import os
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback

# 导入底层环境与造物主
from gymnasium_env.envs.ant_env_v5 import AntEnv
from ant_env_test.co_evolution_main import build_mutant_ant_xml

# 导入 GCN 神经接头与大脑架构
from ant_env_test.graph_obs_wrapper import AntGraphObsWrapper, AntActionWrapper
# 注意：请确保你之前写的 MaskedGraphSACPolicy 等代码在 masked_policy.py 中
from ant_env_test.masked_policy import MaskedGraphSACPolicy 
from stable_baselines3.common.monitor import Monitor
# ==========================================
# ⚙️ 炼丹炉全局配置
# ==========================================
NUM_LEGS = 6              # 钦定 6 足对称海星作为完美宿主
MAX_LEGS_UNIVERSE = 8     # 宇宙法则：未来进化最大支持 8 条腿，必须在这里锁死维度！
TOTAL_TIMESTEPS = 1_000_000 # 100万步，足够 GCN 摸透物理法则
LOG_DIR = "./tb_logs/locomotion_pretrain_v0"
CKPT_DIR = "./checkpoints/locomotion_base_brain_test_v0"

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)


def generate_standard_genome(num_legs, thigh_len=0.2, calf_len=0.2):
    """
    生成绝对对称、标准尺寸的基线基因组，用于预训练打底。
    """
    standard_genome = []
    for i in range(num_legs):
        angle = (360 / num_legs) * i
        # 预训练时的标准关节活动范围
        hip_min, hip_max = -30.0, 30.0
        ankle_min, ankle_max = 30.0, 70.0
        
        leg_gene = [angle, thigh_len, calf_len, hip_min, hip_max, ankle_min, ankle_max]
        standard_genome.append(leg_gene)
        
    return standard_genome

def make_pretrain_env(render_mode=None):
    """🛠️ 组装初代 6 足标准训练舱"""
    TEMP_XML_DIR = "./temp_xmls"
    os.makedirs(TEMP_XML_DIR, exist_ok=True)
    xml_path = os.path.join(TEMP_XML_DIR, f"pretrain_ant_{NUM_LEGS}legs.xml")
    genome = generate_standard_genome(num_legs=NUM_LEGS, thigh_len=0.2, calf_len=0.2)
    # 生成标准的、对称的、健康的初代身体
    xml_str = build_mutant_ant_xml(genome)
    with open(xml_path, "w") as f:
        f.write(xml_str)
        
    def _init():
        # env = AntEnv(xml_file=xml_path, render_mode=render_mode)
        # env = AntEnv(xml_file=xml_path, render_mode=render_mode, reset_noise_scale=0.0)
        env = AntEnv(
            xml_file=xml_path, 
            render_mode=render_mode, 
            reset_noise_scale=0.0,
            healthy_reward=0.05,         # 📉 狂砍底薪：站着不动的工资从 0.5 降到 0.05
            forward_reward_weight=3.0    # 📈 重赏勇夫：只要产生向前的 X 轴速度，奖励翻 3 倍！
        )
        env = gym.wrappers.TimeLimit(env, max_episode_steps=1000) 
        
        # 🕸️ 图网络解析器
        env = AntGraphObsWrapper(env, num_legs=NUM_LEGS, max_legs=MAX_LEGS_UNIVERSE)
        env = AntActionWrapper(env, num_legs=NUM_LEGS, max_legs=MAX_LEGS_UNIVERSE)
        
        # 🎙️ 新增：套上 SB3 的监听器，记录分数和寿命！
        env = Monitor(env) 
        
        return env
    
    return _init

if __name__ == "__main__":
    print(f"\n🔥 正在构建初代 {NUM_LEGS} 足标准母体...")
    
    # 实例化向量化环境
    train_env = DummyVecEnv([make_pretrain_env(render_mode=None)])
    
    print("🧠 正在将通用图卷积神经网络 (GCN) 植入母体...")
    
    # 挂载我们自定义的 GCN 策略
    model = SAC(
        policy=MaskedGraphSACPolicy,
        env=train_env,
        learning_rate=3e-4,
        buffer_size=100_000,   # 显存够的话可以开到 1_000_000
        batch_size=256,
        ent_coef='auto',       # 自动调节熵，鼓励早期探索
        gamma=0.99,
        tau=0.005,
        tensorboard_log=LOG_DIR,
        device="auto",         # 自动调用 GPU/MPS
        verbose=1
    )
    
    # 设置存档点，每 10 万步保存一次脑切片，防止意外断电
    checkpoint_callback = CheckpointCallback(
        save_freq=100_000, 
        save_path=CKPT_DIR,
        name_prefix="gcn_locomotion_base"
    )

    print(f"\n🚀 点火！开始 {TOTAL_TIMESTEPS} 步的高强度预训练...")
    print("   建议新开一个终端运行: tensorboard --logdir=./tb_logs")
    
    # 开启无尽的奔跑与跌倒循环
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS, 
        callback=checkpoint_callback,
        log_interval=10 # 每 10 个回合打印一次日志
    )
    
    # 保存终极完美火种
    final_model_path = os.path.join(CKPT_DIR, "locomotion_base_brain_final")
    model.save(final_model_path)
    print(f"\n🎉 预训练圆满结束！火种已保存至: {final_model_path}.zip")
    
    train_env.close()