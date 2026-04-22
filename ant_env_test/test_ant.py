import os
import time
import numpy as np
import gymnasium as gym

# 导入我们亲手打造的四大核心组件
from gymnasium_env.envs.ant_env_v5 import AntEnv
from ant_env_test.graph_obs_wrapper import AntGraphObsWrapper, AntActionWrapper
from ant_env_test.co_evolution_main import build_mutant_ant_xml 

def test_mutant_environment(num_legs=5):
    print(f"\n🧬 正在培养 {num_legs} 足异形测试体...")
    
    # 1. 动态生成物理世界图纸
    TEMP_XML_DIR = "./temp_xmls"
    os.makedirs(TEMP_XML_DIR, exist_ok=True)
    xml_path = os.path.join(TEMP_XML_DIR, f"test_ant_{num_legs}legs.xml")
    
    xml_str = build_mutant_ant_xml(num_legs=num_legs, thigh_len=0.2, calf_len=0.2)
    with open(xml_path, "w") as f:
        f.write(xml_str)
        
    # 2. 实例化物理容器，并开启人类观察模式
    env = AntEnv(xml_file=xml_path, render_mode="human")
    env = gym.wrappers.TimeLimit(env, max_episode_steps=500)
    
    # 🕸️ 植入神经网络接头：套上我们的图网络解析器
    env = AntGraphObsWrapper(env, num_legs=num_legs, max_legs=8)
    env = AntActionWrapper(env, num_legs=num_legs, max_legs=8)
    
    # 3. 唤醒世界
    obs, info = env.reset()
    
    # --------------------------------------------------
    # 📊 维度核对大检阅！(这是 GCN 能够完美运转的生命线)
    # --------------------------------------------------
    print("\n" + "="*50)
    print("✅ 环境初始化成功！正在核对 GCN 拓扑图接口维度：")
    print(f"  🧠 [Nodes Shape]:  {obs['nodes'].shape}  (预期: 17x4 -> 1躯干+16关节，每个节点4个特征)")
    print(f"  🎭 [Mask Shape]:   {obs['mask'].shape}   (预期: 17 -> 即最大节点容量)")
    print(f"  🌍 [Global Shape]: {obs['global'].shape}    (预期: 6 -> 躯干的前庭觉与速度特征)")
    print(f"  💪 [Action Space]: {env.action_space.shape}    (预期: {num_legs * 2} -> 当前存在的真实马达数)")
    
    # 打印 Mask，直观确认是否正确屏蔽了未发育出来的腿
    print(f"  🔍 [有效 Mask]:    {obs['mask']}")
    print("="*50 + "\n")
    
    time.sleep(2) # 停顿 2 秒，让你有时间看清控制台的自检报告
    
    # 4. 暴力随机驾驶测试 (Dry Run / Fuzzing)
    print("🌪️ 开启随机动作狂暴测试，准备切入 MuJoCo 渲染视窗...")
    total_reward = 0.0
    
    for step in range(500):
        # 产生当前动作空间的合法随机动作 [-1.0, 1.0]
        random_action = env.action_space.sample() 
        
        obs, reward, terminated, truncated, info = env.step(random_action)
        total_reward += reward
        
        # 降速渲染：让画面慢一点，方便肉眼观察关节运动是否穿模、触地反馈是否正常
        time.sleep(0.02) 
        
        if terminated or truncated:
            reason = "💀 失去平衡/物理崩溃 (Terminated)" if terminated else "⏱️ 寿命耗尽 (Truncated)"
            print(f"   [Step {step}] 异形停止活动。原因: {reason}")
            print(f"   [最终状态] Z轴高度: {obs['global'][0]:.3f}, 累计存活得分: {total_reward:.2f}")
            
            # 满血复活，继续测试
            obs, info = env.reset()
            total_reward = 0.0
            print("⚡ 异形已在随机扰动下重生，继续下一次跌倒测试...")

    env.close()
    print("\n🎉 物理环境干跑测试圆满结束！如果没有报 NaN 或越界错误，说明系统坚不可摧！")

if __name__ == "__main__":
    # 故意传入一个 5 足这样的非对称奇数，最能考验系统的鲁棒性
    test_mutant_environment(num_legs=5)