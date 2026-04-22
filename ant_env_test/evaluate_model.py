import os
import time
import gymnasium as gym
from stable_baselines3 import SAC

# 导入底层环境与 GCN 组件
from gymnasium_env.envs.ant_env_v5 import AntEnv
from ant_env_test.co_evolution_main import build_mutant_ant_xml
from ant_env_test.graph_obs_wrapper import AntGraphObsWrapper, AntActionWrapper
from ant_env_test.masked_policy import MaskedGraphSACPolicy

# ==========================================
# 🧬 评测特供：标准基因克隆舱
# ==========================================
def generate_standard_genome(num_legs, thigh_len=0.2, calf_len=0.2):
    """
    生成绝对对称、标准尺寸的基线基因组，用于测试基座大脑的标准表现。
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
# ⚖️ 阅兵场主程序
# ==========================================
def evaluate_brain(model_path, num_legs=6):
    print(f"\n🧠 正在将冷冻大脑注入 {num_legs} 足躯体...")
    
    # 1. 组装物理测试舱
    TEMP_XML_DIR = "./temp_xmls"
    VIDEO_DIR = "./eval_videos" # 🎥 新增：视频输出目录
    
    os.makedirs(TEMP_XML_DIR, exist_ok=True)
    os.makedirs(VIDEO_DIR, exist_ok=True)
    
    xml_path = os.path.join(TEMP_XML_DIR, f"eval_ant_{num_legs}legs.xml")
    
    # 🚨 核心修改 1：纯基因驱动渲染
    standard_genome = generate_standard_genome(num_legs=num_legs)
    xml_str = build_mutant_ant_xml(standard_genome)
    
    with open(xml_path, "w") as f:
        f.write(xml_str)
        
    # 2. 实例化环境 (关闭人类观察，开启像素阵列输出)
    env = AntEnv(
        xml_file=xml_path, 
        render_mode="rgb_array",        # 🎥 修改：必须为 rgb_array 才能录制视频
        reset_noise_scale=0.0,          
        healthy_z_range=(0.13, 1.5), 
        healthy_reward=1.0,             
        forward_reward_weight=3.0,   
        ctrl_cost_weight=0.05           
    )
    
    # 🎥 新增：套上录像包装器
    # name_prefix 包含了腿的数量，这样批量测试不同构型时视频不会被覆盖
    env = gym.wrappers.RecordVideo(
        env, 
        video_folder=VIDEO_DIR, 
        name_prefix=f"ant_{num_legs}legs",
        episode_trigger=lambda x: True  # 记录每一个回合
    )
    
    # 测试时给它无限长的时间
    env = gym.wrappers.TimeLimit(env, max_episode_steps=2000) 
    
    # 套上图网络解析器 (宇宙法则最大 8 足)
    env = AntGraphObsWrapper(env, num_legs=num_legs, max_legs=8)
    env = AntActionWrapper(env, num_legs=num_legs, max_legs=8)
    
    # 3. 唤醒 GCN 大脑
    print(f"📥 加载权重: {model_path}")
    model = SAC.load(model_path, env=env, custom_objects={'policy_class': MaskedGraphSACPolicy})
    
    # 4. 跑圈测试
    print(f"🎬 演习开始！视频正在后台渲染并保存至 {VIDEO_DIR} ...")
    obs, info = env.reset()
    
    for episode in range(5): # 测试 5 个回合
        total_reward = 0.0
        step_count = 0
        while True:
            # deterministic=True 代表使用大脑最确信的动作，不加随机噪音
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            
            total_reward += reward
            step_count += 1
            
            # 🎥 移除 time.sleep(0.02)
            # 在 rgb_array 模式下，系统会全速渲染并编码，不需要人工降速
            
            if terminated or truncated:
                reason = "💀 摔倒翻车" if terminated else "⏱️ 坚持到了最后"
                print(f"   [回合 {episode+1}] 结束 | 存活: {step_count} 步 | 总得分: {total_reward:.2f} | 原因: {reason}")
                obs, info = env.reset()
                break

    env.close() # 🎥 极其重要：必须 close，否则最后一段视频的 mp4 封装可能会损坏
    print("\n🎉 阅兵结束！")

if __name__ == "__main__":
    MODEL_PATH = "./checkpoints/generalist_base_brain_normal/gcn_generalist_base_1400000_steps" 
    
    if os.path.exists(MODEL_PATH + ".zip"):
        evaluate_brain(MODEL_PATH, num_legs=5)
    else:
        print(f"❌ 找不到模型文件: {MODEL_PATH}.zip。请先等训练脚本跑出存档！")