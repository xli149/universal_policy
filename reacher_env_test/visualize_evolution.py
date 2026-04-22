import os
import time
import gymnasium as gym
from stable_baselines3 import SAC
from gymnasium.wrappers import TimeLimit

# 导入你的自定义环境和图神经网络包装器
import gymnasium_env
from graph_obs_wrapper import PaddedGraphObsWrapper, PaddedActionWrapper

# ==========================================
# ⚙️ 目录配置
# ==========================================
XML_DIR = "./checkpoints_obstacle/evolution_milestones"
BRAIN_DIR = "./checkpoints_obstacle/co_evolution"
PRETRAINED_BRAIN = "./checkpoints/pretrained_base_brain_v1/pretrained_base_brain.zip"
MAX_JOINTS = 10
MAX_STEPS = 100

def list_files(directory, extension):
    if not os.path.exists(directory):
        return []
    files = [f for f in os.listdir(directory) if f.endswith(extension)]
    # 按生成时间或代数简单排序
    files.sort()
    return files

def get_n_joints_from_xml(xml_name):
    # 从类似 'RECORD_gen15_best_4j_0.10_0.05.xml' 中提取 '4'
    parts = xml_name.split('_')
    for part in parts:
        if part.endswith('j') and part[:-1].isdigit():
            return int(part[:-1])
    # 默认 fallback
    return 1

def run_visualization():
    print("\n" + "="*50)
    print("🤖 进化全息投影仪 - 跨代交叉测试平台")
    print("="*50)

    # 1. 挑选肉体 (XML)
    xml_files = list_files(XML_DIR, ".xml")
    if not xml_files:
        print(f"🚨 找不到任何化石！请确保 {XML_DIR} 下有 .xml 文件。")
        return

    print("\n🦴 第一步：请选择你要测试的【肉体图纸】")
    for i, f in enumerate(xml_files):
        print(f"  [{i}] {f}")
    xml_idx = int(input("👉 输入肉体编号: "))
    selected_xml = xml_files[xml_idx]
    xml_path = os.path.join(XML_DIR, selected_xml)
    n_joints = get_n_joints_from_xml(selected_xml)

    # 2. 挑选灵魂 (Brain .zip)
    brain_files = list_files(BRAIN_DIR, ".zip")
    print("\n🧠 第二步：请选择你要注入的【神经网络灵魂】")
    print(f"  [-1] 👶 零基础初代大脑 (Pretrained Base Brain)")
    for i, f in enumerate(brain_files):
        print(f"  [{i}] {f}")
    
    brain_idx = int(input("👉 输入大脑编号: "))
    if brain_idx == -1:
        brain_path = PRETRAINED_BRAIN
    else:
        brain_path = os.path.join(BRAIN_DIR, brain_files[brain_idx])

    if not os.path.exists(brain_path):
        print(f"🚨 找不到大脑文件: {brain_path}")
        return

    print(f"\n🚀 组合完毕！正在将 {os.path.basename(brain_path)} 注入 {selected_xml}...")
    
    # 3. 构建可视化环境
    env = gym.make("gymnasium_env/Reacher2D-v0", xml_file=xml_path, render_mode="human")
    env = TimeLimit(env, max_episode_steps=MAX_STEPS)
    env = PaddedGraphObsWrapper(env, max_joints=MAX_JOINTS, n_arm_joints=n_joints)
    env = PaddedActionWrapper(env, max_joints=MAX_JOINTS, n_arm_joints=n_joints)

    # 4. 加载模型
    model = SAC.load(brain_path, env=env)

    # 5. 开始物理模拟演示
    try:
        for episode in range(5):  # 演示 5 个回合
            obs, _ = env.reset()
            terminated = truncated = False
            total_reward = 0.0
            
            print(f"\n🎬 回合 {episode + 1}/5 开始...")
            
            while not (terminated or truncated):
                # 开启 deterministic=True 以展示模型真正的控制水平，不加入探索噪音
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                
                # 🐌 强制减速，让你能用肉眼看清动作细节 (约 50 FPS)
                time.sleep(0.02)
                
            print(f"🎯 回合结束 | 累计得分: {total_reward:.2f}")
            time.sleep(1) # 回合间歇停顿一下
            
    except KeyboardInterrupt:
        print("\n⏹️ 用户手动终止了可视化。")
    finally:
        env.close()

if __name__ == "__main__":
    run_visualization()