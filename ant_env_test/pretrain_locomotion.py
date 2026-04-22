import os
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor

# 导入底层环境
from gymnasium_env.envs.ant_env_v5 import AntEnv
# 🚨 导入你刚才写好的终极造物主！
from ant_env_test.co_evolution_main import build_mutant_ant_xml

# 导入 GCN 神经接头与大脑架构
from ant_env_test.graph_obs_wrapper import AntGraphObsWrapper, AntActionWrapper
from ant_env_test.masked_policy import MaskedGraphSACPolicy 

# ==========================================
# ⚙️ “泛化大师”炼丹炉全局配置
# ==========================================
# 我们开启 4 个平行宇宙，分别放入 4, 5, 6, 8 条腿的标准生物
LEG_VARIANTS = [4, 5, 6, 7]  
NUM_ENVS = len(LEG_VARIANTS) 

MAX_LEGS_UNIVERSE = 8       
TOTAL_TIMESTEPS = 2_000_000 
LOG_DIR = "./tb_logs/generalist_pretrain_normal"
CKPT_DIR = "./checkpoints/generalist_base_brain_normal"

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)

# ==========================================
# 🧬 预训练特供：标准基因克隆舱
# ==========================================
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

# ==========================================
# 🌍 平行宇宙构建器
# ==========================================
def make_pretrain_env(env_rank, num_legs, render_mode=None):
    """🛠️ 组装变异训练舱：为每个平行宇宙生成特定的 XML"""
    TEMP_XML_DIR = "./temp_xmls"
    os.makedirs(TEMP_XML_DIR, exist_ok=True)
    xml_path = os.path.join(TEMP_XML_DIR, f"pretrain_ant_rank{env_rank}_{num_legs}legs.xml")
    
    # 🚨 核心衔接：先生成标准基因，再喂给终极造物主！
    standard_genome = generate_standard_genome(num_legs=num_legs)
    xml_str = build_mutant_ant_xml(standard_genome)
    
    with open(xml_path, "w") as f:
        f.write(xml_str)
        
    def _init():
        env = AntEnv(
            xml_file=xml_path, 
            render_mode=render_mode, 
            reset_noise_scale=0.0,
            # 物理法则与 GA 演化脚本保持绝对一致！
            healthy_z_range=(0.13, 1.5), 
            healthy_reward=0.05,          
            forward_reward_weight=3.0,   
            # ctrl_cost_weight=0.05        
        )
        env = gym.wrappers.TimeLimit(env, max_episode_steps=1000) 
        
        # 🕸️ 图网络解析器
        env = AntGraphObsWrapper(env, num_legs=num_legs, max_legs=MAX_LEGS_UNIVERSE)
        env = AntActionWrapper(env, num_legs=num_legs, max_legs=MAX_LEGS_UNIVERSE)
        
        env = Monitor(env) 
        return env
    
    return _init

# ==========================================
# 🚀 炼丹炉主程序
# ==========================================
if __name__ == "__main__":
    print(f"\n🔥 正在撕裂空间，准备开启 {NUM_ENVS} 个不同形态的平行宇宙...")
    print(f"🌍 宇宙形态分布: {LEG_VARIANTS} 足")
    
    # 启动多进程并行收集经验
    env_fns = [lambda r=rank, l=legs: make_pretrain_env(r, l)() for rank, legs in enumerate(LEG_VARIANTS)]
    train_env = SubprocVecEnv(env_fns)
    
    print("🧠 正在将通用图卷积神经网络 (GCN) 植入所有母体...")
    
    model = SAC(
        policy=MaskedGraphSACPolicy,
        env=train_env,
        learning_rate=3e-4,
        buffer_size=200_000,
        batch_size=256,
        ent_coef='auto',       
        gamma=0.99,
        tau=0.005,
        tensorboard_log=LOG_DIR,
        device="auto",         
        verbose=1
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=max(100_000 // NUM_ENVS, 1), 
        save_path=CKPT_DIR,
        name_prefix="gcn_generalist_base"
    )

    print(f"\n🚀 点火！开始 {TOTAL_TIMESTEPS} 步的形态大一统 (MDR) 预训练...")
    print("   👉 建议新开一个终端运行: tensorboard --logdir=./tb_logs/generalist_pretrain")
    
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS, 
        callback=checkpoint_callback,
        log_interval=4 
    )
    
    final_model_path = os.path.join(CKPT_DIR, "gcn_generalist_base_final")
    model.save(final_model_path)
    print(f"\n🎉 泛化大师预训练圆满结束！火种已保存至: {final_model_path}.zip")
    
    train_env.close()