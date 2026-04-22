import os
import time
import gymnasium as gym
from stable_baselines3 import SAC

# 导入底层环境与 GCN 组件
from gymnasium_env.envs.ant_env_v5 import AntEnv
from ant_env_test.co_evolution_main import build_mutant_ant_xml
from ant_env_test.graph_obs_wrapper import AntGraphObsWrapper, AntActionWrapper
from ant_env_test.masked_policy import MaskedGraphSACPolicy

def evaluate_brain(model_path, num_legs=6):
    print(f"\n🧠 正在将冷冻大脑注入 {num_legs} 足躯体...")
    
    # 1. 组装同样的物理测试舱
    TEMP_XML_DIR = "./temp_xmls"
    os.makedirs(TEMP_XML_DIR, exist_ok=True)
    xml_path = os.path.join(TEMP_XML_DIR, f"eval_ant_{num_legs}legs.xml")
    
    xml_str = build_mutant_ant_xml(num_legs=num_legs, thigh_len=0.2, calf_len=0.2)
    with open(xml_path, "w") as f:
        f.write(xml_str)
        
    # 2. 实例化环境 (开启人类观察模式)
    env = AntEnv(
            xml_file=xml_path, 
            render_mode="human",
            # healthy_z_range=(0.18, 1.5),  # 扩宽健康区间
            healthy_z_range=(0.13, 1.5),
            reset_noise_scale=0.05,        # 降低初始随机噪音
            # 👇 核心：加入这行代码！让镜头死死锁住机器人躯干
            # default_camera_config={
            #     "trackbodyid": 1,     # 追踪身体的 ID，1 通常就是第一个 body（torso 躯干）
            #     "distance": 4.0,      # 镜头距离（缩放）
            #     "elevation": -90.0,   # 俯仰角，-20度刚好可以舒服地俯视它的腿部动作
            #     "azimuth": 90.0       # 方位角，从侧后方观察
            # }
        )
    # 测试时给它无限长的时间
    env = gym.wrappers.TimeLimit(env, max_episode_steps=2000) 
    
    # 套上图网络解析器 (宇宙法则最大 8 足)
    env = AntGraphObsWrapper(env, num_legs=num_legs, max_legs=8)
    env = AntActionWrapper(env, num_legs=num_legs, max_legs=8)
    
    # 3. 唤醒 GCN 大脑
    print(f"📥 加载权重: {model_path}")
    # 这里必须传入 custom_objects 确保自定义的 Policy 能被正确解析
    model = SAC.load(model_path, env=env, custom_objects={'policy_class': MaskedGraphSACPolicy})
    
    # 4. 跑圈测试
    print("🎬 演习开始！请移步 MuJoCo 窗口观看...")
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
            time.sleep(0.02) # 降速播放，方便看清动作
            
            if terminated or truncated:
                reason = "💀 摔倒翻车" if terminated else "⏱️ 坚持到了最后"
                print(f"   [回合 {episode+1}] 结束 | 存活: {step_count} 步 | 总得分: {total_reward:.2f} | 原因: {reason}")
                obs, info = env.reset()
                break

    env.close()
    print("\n🎉 阅兵结束！")

if __name__ == "__main__":
    # ⚠️ 等你的炼丹炉保存了 checkpoint 后，把下面的路径替换成真实的 zip 文件路径
    # 例如: "./checkpoints/locomotion_base_brain/gcn_locomotion_base_100000_steps"
    # 注意：路径不要带 .zip 后缀，SB3 会自己找
    
    MODEL_PATH = "./checkpoints/locomotion_base_brain/gcn_locomotion_base_900000_steps" 
    
    if os.path.exists(MODEL_PATH + ".zip"):
        evaluate_brain(MODEL_PATH, num_legs=6)
    else:
        print(f"❌ 找不到模型文件: {MODEL_PATH}.zip。请先等训练脚本跑出存档！")