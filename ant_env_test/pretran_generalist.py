import os
import gymnasium as gym
from stable_baselines3 import SAC
# 🚨 核心修改 1：引入多进程并行环境
from stable_baselines3.common.vec_env import SubprocVecEnv 
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from gymnasium_env.envs.ant_env_v5 import AntEnv

# 🚨 核心修改 2：导入终极造物主和随机基因生成器
from ant_env_test.co_evolution_main import build_asymmetric_ant_xml
from ant_env_test.ga_evolution_loop import generate_random_genome

from ant_env_test.graph_obs_wrapper import AntGraphObsWrapper, AntActionWrapper
from ant_env_test.masked_policy import MaskedGraphSACPolicy 

# ==========================================
# ⚙️ “泛化大师”炼丹炉全局配置
# ==========================================
NUM_ENVS = 8              # 🌍 平行宇宙数量 (建议设为 CPU 核心数，比如 8 或 16)
MAX_LEGS_UNIVERSE = 8     # 宇宙法则：最大支持 8 条腿
TOTAL_TIMESTEPS = 2_000_000 # 📈 难度剧增，训练步数翻倍到 200万步！
LOG_DIR = "./tb_logs/generalist_pretrain"
CKPT_DIR = "./checkpoints/generalist_base_brain"

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)

def make_diverse_env(env_rank):
    """🛠️ 组装充满未知的变异训练舱"""
    def _init():
        # 1. 每次初始化平行宇宙时，随机生成一个基因！(4到8条腿，长短不一)
        random_genome = generate_random_genome(min_leg=4, max_leg=8)
        num_legs = len(random_genome)
        
        # 2. 为当前进程生成独占的 XML 文件，防止多进程读写冲突
        TEMP_XML_DIR = "./temp_xmls"
        os.makedirs(TEMP_XML_DIR, exist_ok=True)
        xml_path = os.path.join(TEMP_XML_DIR, f"pretrain_env_rank_{env_rank}.xml")
        
        xml_str = build_asymmetric_ant_xml(random_genome)
        with open(xml_path, "w") as f:
            f.write(xml_str)
            
        # 3. 实例化物理环境 (🚨 注入我们刚刚讨论的“严苛物理法则”！)
        env = AntEnv(
            xml_file=xml_path, 
            render_mode=None, 
            reset_noise_scale=0.0,
            healthy_reward=1.0,           # 💰 提高底薪，鼓励努力活着
            healthy_z_range=(0.25, 1.0),  # 💀 抬高死亡线至 0.25，严禁“翻面乌龟”躺平！
            forward_reward_weight=3.0     # 🏃 重赏向前奔跑的行为
        )
        env = gym.wrappers.TimeLimit(env, max_episode_steps=1000) 
        
        # 4. 图网络解析器 (不管几条腿，全部 Pad 到 8 条腿的维度)
        env = AntGraphObsWrapper(env, num_legs=num_legs, max_legs=MAX_LEGS_UNIVERSE)
        env = AntActionWrapper(env, num_legs=num_legs, max_legs=MAX_LEGS_UNIVERSE)
        env = Monitor(env) 
        
        return env
    
    return _init

if __name__ == "__main__":
    print(f"\n🔥 正在撕裂空间，开启 {NUM_ENVS} 个形态各异的平行宇宙...")
    
    # 🚨 核心修改 3：启动多进程向量化环境
    train_env = SubprocVecEnv([make_diverse_env(i) for i in range(NUM_ENVS)])
    
    print("🧠 正在将通用图卷积神经网络 (GCN) 植入所有母体...")
    
    # 挂载自定义的 GCN 策略
    model = SAC(
        policy=MaskedGraphSACPolicy,
        env=train_env,
        learning_rate=3e-4,
        buffer_size=300_000,   # 因为形态变多了，经验池适当开大一点
        batch_size=256,
        ent_coef='auto',       
        gamma=0.99,
        tau=0.005,
        tensorboard_log=LOG_DIR,
        device="auto",         
        verbose=1
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=max(100_000 // NUM_ENVS, 1), # 适配多线程的保存频率
        save_path=CKPT_DIR,
        name_prefix="gcn_generalist_base"
    )

    print(f"\n🚀 点火！开始 {TOTAL_TIMESTEPS} 步的形态域随机化 (MDR) 预训练...")
    print("   建议新开一个终端运行: tensorboard --logdir=./tb_logs/generalist_pretrain")
    
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS, 
        callback=checkpoint_callback,
        log_interval=4 
    )
    
    final_model_path = os.path.join(CKPT_DIR, "gcn_generalist_base_final")
    model.save(final_model_path)
    print(f"\n🎉 泛化大师预训练圆满结束！火种已保存至: {final_model_path}.zip")
    
    train_env.close()