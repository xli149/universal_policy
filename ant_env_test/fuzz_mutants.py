import os
import random
import time
import numpy as np
import gymnasium as gym

# 导入你的环境和 XML 生成器
from gymnasium_env.envs.ant_env_v5 import AntEnv
from ant_env_test.co_evolution_main import build_asymmetric_ant_xml
from ant_env_test.graph_obs_wrapper import AntGraphObsWrapper, AntActionWrapper

def generate_random_genome(max_legs=8):
    """🎲 疯狂掷骰子：生成绝对随机的非对称基因"""
    num_legs = random.randint(3, max_legs)
    genome = []
    
    for _ in range(num_legs):
        angle = random.uniform(0, 360)
        thigh_len = random.uniform(0.05, 0.4) 
        calf_len = random.uniform(0.05, 0.4)
        hip_min = random.uniform(-60, 0)
        hip_max = random.uniform(0, 60)
        ankle_min = random.uniform(-90, 0)
        ankle_max = random.uniform(0, 90)
        
        genome.append([angle, thigh_len, calf_len, hip_min, hip_max, ankle_min, ankle_max])
    return genome

def fuzz_and_visualize(test_rounds=1000, sample_size=10):
    print(f"🌪️ [第一阶段] 开始静默 Fuzzing 测试，目标生成 {test_rounds} 种随机异形...\n")
    
    TEMP_XML_DIR = "./temp_fuzz_xmls"
    os.makedirs(TEMP_XML_DIR, exist_ok=True)
    
    successful_genomes = []
    fatal_genomes = []

    # ==========================================
    # 🕵️‍♂️ 第一阶段：无头模式 (Headless) 极速筛选
    # ==========================================
    for i in range(test_rounds):
        # 进度条
        percent = (i + 1) / test_rounds
        bar_length = 40
        filled_len = int(bar_length * percent)
        bar = '█' * filled_len + '-' * (bar_length - filled_len)
        print(f"\r🚀 极速测试中: [{bar}] {i+1}/{test_rounds} ({percent*100:.1f}%)", end="", flush=True)
        
        genome = generate_random_genome()
        xml_path = os.path.join(TEMP_XML_DIR, f"fuzz_ant_{i}.xml")
        xml_str = build_asymmetric_ant_xml(genome)
        
        with open(xml_path, "w") as f:
            f.write(xml_str)
            
        try:
            # 静默加载：不传 render_mode
            env = AntEnv(xml_file=xml_path)
            obs, _ = env.reset()
            
            # 演算 20 步，测试物理碰撞是否会引发张量爆炸
            for _ in range(20):
                action = env.action_space.sample() 
                obs, reward, terminated, truncated, info = env.step(action)
                
                if np.isnan(obs['qpos']).any() or np.isnan(obs['qvel']).any():
                    raise ValueError("物理引擎张量爆炸 (NaN)")
                
                if terminated or truncated:
                    break
                    
            env.close()
            # 存活下来的基因，加入精英库
            successful_genomes.append(genome)
            
        except Exception as e:
            fatal_genomes.append((genome, str(e)))
            
        finally:
            if os.path.exists(xml_path):
                os.remove(xml_path)

    print(f"\n\n✅ Fuzz 测试结束！存活率: {len(successful_genomes)} / {test_rounds}")

    # ==========================================
    # 📺 第二阶段：斗兽场模式 (抽样渲染展示)
    # ==========================================
    if not successful_genomes:
        print("💀 全军覆没，没有存活的基因可以展示。请检查生成器或环境参数！")
        return

    # 从存活库中随机抽取 sample_size 个（如果存活数不够，就展示全部存活的）
    actual_sample_size = min(sample_size, len(successful_genomes))
    sampled_genomes = random.sample(successful_genomes, actual_sample_size)

    print(f"\n🎬 [第二阶段] 从存活库中随机抽取 {actual_sample_size} 个异形进行可视化展示！")
    print("⚠️  请将视线移至弹出的 MuJoCo 渲染窗口...\n")

    for idx, genome in enumerate(sampled_genomes):
        num_legs = len(genome)
        print(f"   ✨ 正在展示异形 {idx+1}/{actual_sample_size} (拥有 {num_legs} 条非对称腿)")
        
        xml_path = os.path.join(TEMP_XML_DIR, "render_temp.xml")
        xml_str = build_asymmetric_ant_xml(genome)
        with open(xml_path, "w") as f:
            f.write(xml_str)
            
        try:
            # 开启 human 渲染模式
            env = AntEnv(xml_file=xml_path, render_mode="human")
            
            # 顺便套上 Wrapper 测一下 Action Space 维度对不对
            env = AntGraphObsWrapper(env, num_legs=num_legs, max_legs=8)
            env = AntActionWrapper(env, num_legs=num_legs, max_legs=8)
            
            env.reset()
            
            # 给它 150 步的随机动作展示时间 (大概 3-4 秒)
            for _ in range(150):
                action = env.action_space.sample()
                env.step(action)
                time.sleep(0.02) # 降速播放
                
            env.close()
            
        except Exception as e:
            print(f"   🚨 渲染报错: {e}")
        finally:
            if os.path.exists(xml_path):
                os.remove(xml_path)

    print("\n🎉 抽样阅兵完毕！如果这些畸形种都没有报错，我们的物理宇宙就彻底稳了！")

if __name__ == "__main__":
    # 执行 1000 轮盲测，然后抽选 10 个出来溜溜
    fuzz_and_visualize(test_rounds=1000, sample_size=10)