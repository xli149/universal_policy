import os
import time
import random
import numpy as np
import gymnasium as gym

# 1. 导入你的本地魔改版环境
from gymnasium_env.envs.ant_env_v5 import AntEnv

# 2. 导入我们的核心组件
# 确保你已经把 build_asymmetric_ant_xml 追加到了 co_evolution_main.py 中
from ant_env_test.co_evolution_main import build_asymmetric_ant_xml 
from ant_env_test.graph_obs_wrapper import AntGraphObsWrapper, AntActionWrapper

def generate_wild_genome(max_legs=8):
    """🎲 疯狂掷骰子：生成绝对随机、非对称、充满魔幻色彩的基因"""
    num_legs = random.randint(3, max_legs) # 至少3条腿，看起来更怪异
    genome = []
    
    # 定义基因突变范围（可以允许一些极端参数，看热闹不嫌大）
    for _ in range(num_legs):
        angle = random.uniform(0, 360)             # 附着角度：全方位随机
        thigh_len = random.uniform(0.1, 0.5)       # 大腿长：从短腿到长腿
        calf_len = random.uniform(0.1, 0.5)        # 小腿长
        hip_min = random.uniform(-40, -10)         # 髋关节活动上限
        hip_max = random.uniform(10, 40)          # 髋关节活动下限
        ankle_min = random.uniform(10, 40)         # 踝关节活动下限 (正常形态)
        ankle_max = random.uniform(60, 100)        # 踝关节活动上限 (限制最大弯曲度)
        
        genome.append([angle, thigh_len, calf_len, hip_min, hip_max, ankle_min, ankle_max])
    return genome

def show_random_creatures(num_to_show=5, duration_per_creature=5):
    """👑 依次生成并可视化随机生物"""
    print(f"🎬 准备展示 {num_to_show} 个随机生成的数字生命形态...")
    print("⚠️  MuJoCo 窗口弹出后，你可以用鼠标拖拽视点，滚轮缩放。")
    
    TEMP_VIS_DIR = "./temp_vis_xmls"
    os.makedirs(TEMP_VIS_DIR, exist_ok=True)
    
    for i in range(num_to_show):
        print(f"\n✨ 正在生成第 {i+1}/{num_to_show} 个异形形态...")
        
        # 1. 生成基因并打印 (看看长啥样)
        genome = generate_wild_genome()
        num_legs = len(genome)
        print(f"   🧬 基因型：{num_legs} 条腿，非对称设计。")
        
        # 2. 生成 XML
        xml_path = os.path.join(TEMP_VIS_DIR, f"vis_ant_{i}.xml")
        xml_str = build_asymmetric_ant_xml(genome)
        with open(xml_path, "w") as f:
            f.write(xml_str)
            
        try:
            # 3. 实例化物理环境 (必须开启 human 模式)
            env = AntEnv(xml_file=xml_path, render_mode="human")
            env = gym.wrappers.TimeLimit(env, max_episode_steps=1000)
            
            # 虽然只是观看，但为了保证 Action Space 维度正确，依然套上 Wrapper
            # Wrapper 这里 max_legs 设为 8，与上面生成保持一致
            env = AntGraphObsWrapper(env, num_legs=num_legs, max_legs=8)
            env = AntActionWrapper(env, num_legs=num_legs, max_legs=8)
            
            env.reset()
            
            print(f"   📺 正在 MuJoCo 窗口展示，持续 {duration_per_creature} 秒...")
            
            start_time = time.time()
            # 4. 驱动循环：注入随机动作，看看这副身体动起来怎么样
            while time.time() - start_time < duration_per_creature:
                # 产生当前动作空间的合法随机动作 [-1.0, 1.0]
                random_action = env.action_space.sample() 
                
                # 执行动作
                obs, reward, terminated, truncated, _ = env.step(random_action)
                
                # 如果摔倒了，就原体重生，继续展示
                if terminated or truncated:
                    env.reset()
                    
            env.close()
            
            # 展示完毕后，删除临时 XML
            os.remove(xml_path)
            
        except Exception as e:
            print(f"   🚨 这个形态太畸形，导致物理引擎崩溃了！错误: {e}")
            if os.path.exists(xml_path):
                os.remove(xml_path)
            continue

    print("\n🎉 展示结束！造物主，你对这些随机形态满意吗？")

if __name__ == "__main__":
    # 随机挑选 5 个生物进行可视化
    show_random_creatures(num_to_show=5, duration_per_creature=5)