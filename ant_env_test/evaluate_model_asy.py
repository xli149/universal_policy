import os
import time
import gymnasium as gym
from stable_baselines3 import SAC

# 导入底层环境与 GCN 组件
from gymnasium_env.envs.ant_env_v5 import AntEnv
# 🚨 注意：这里改用终极造物主 build_asymmetric_ant_xml
from ant_env_test.co_evolution_main import build_asymmetric_ant_xml
from ant_env_test.graph_obs_wrapper import AntGraphObsWrapper, AntActionWrapper
from ant_env_test.masked_policy import MaskedGraphSACPolicy

def evaluate_mutant_brain(model_path):
    print(f"\n🧠 正在将 6 足冷冻大脑注入【变异残疾躯体】...")
    
    # ==========================================
    # 🧬 战役一：基因编辑 (残疾变异体)
    # ==========================================
    # genome = [角度, 大腿长, 小腿长, 髋关节min, 髋关节max, 踝关节min, 踝关节max]
    # 我们故意只给它 5 条腿 (360/5 = 72度)，相当于砍掉了一条腿！
    # 并且故意把第 1 条腿的关节活动范围缩死，模拟“骨折”或“关节炎”！
    mutant_genome = [
        [0,   0.2, 0.2, -30, 30, 30, 70],  # 正常腿
        [72,  0.2, 0.2, -5,  5,  50, 55],  # 💥 骨折腿：关节几乎被锁死！
        [144, 0.2, 0.2, -30, 30, 30, 70],  # 正常腿
        [216, 0.2, 0.2, -30, 30, 30, 70],  # 正常腿
        [288, 0.2, 0.2, -30, 30, 30, 70],  # 正常腿
    ]
    
    # num_legs = len(mutant_genome)
    num_legs = 7

    
    # 1. 组装物理测试舱
    # TEMP_XML_DIR = "./temp_xmls"
    TEMP_XML_DIR = "./test_unit"
    os.makedirs(TEMP_XML_DIR, exist_ok=True)
    xml_path = os.path.join(TEMP_XML_DIR, f"random_generate_legs_test_parents.xml")
    # /Users/chrislee/Documents/mujoco_test/temp_xmls/eval_ga_ant.xml
    # xml_path = os.path.join(TEMP_XML_DIR, f"random_generate_legs_test_parents.xml")
    # 使用非对称 XML 生成器
    # xml_str = build_asymmetric_ant_xml(mutant_genome)
    # with open(xml_path, "w") as f:
    #     f.write(xml_str)
        
    # 2. 实例化环境
    env = AntEnv(
            xml_file=xml_path, 
            render_mode="human",
            # healthy_z_range=(0.13, 1.5),  # 容忍度调低，允许瘸子趴低一点
            reset_noise_scale=0.0,        # 关掉出生噪音
            # default_camera_config={
            #     "trackbodyid": 1,     
            #     "distance": 4.0,      
            #     "elevation": -30.0,   
            #     "azimuth": 45.0       
            # }
        )
    env = gym.wrappers.TimeLimit(env, max_episode_steps=2000) 
    
    # 🚨 GCN 魔法发挥作用的地方：传入真实的腿数 (5)，最大槽位 (8)
    env = AntGraphObsWrapper(env, num_legs=num_legs, max_legs=8)
    env = AntActionWrapper(env, num_legs=num_legs, max_legs=8)
    
    # 3. 唤醒 GCN 大脑
    print(f"📥 加载权重: {model_path}")
    model = SAC.load(model_path, env=env, custom_objects={'policy_class': MaskedGraphSACPolicy})
    
    # 4. 跑圈测试
    print("🎬 截肢演习开始！请移步 MuJoCo 窗口观看...")
    obs, info = env.reset()
    
    for episode in range(5):
        total_reward = 0.0
        step_count = 0
        while True:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            
            total_reward += reward
            step_count += 1
            time.sleep(0.02) 
            
            if terminated or truncated:
                reason = "💀 摔倒翻车" if terminated else "⏱️ 坚持到了最后"
                print(f"   [回合 {episode+1}] 结束 | 存活: {step_count} 步 | 总得分: {total_reward:.2f} | 原因: {reason}")
                obs, info = env.reset()
                break

    env.close()
    print("\n🎉 变异体测试结束！")

if __name__ == "__main__":
    # 确保路径指向你那个拿了 900+ 高分的大脑！
    MODEL_PATH = "./checkpoints/locomotion_base_brain/gcn_locomotion_base_900000_steps" 
    
    if os.path.exists(MODEL_PATH + ".zip"):
        evaluate_mutant_brain(MODEL_PATH)
    else:
        print(f"❌ 找不到模型文件: {MODEL_PATH}.zip")