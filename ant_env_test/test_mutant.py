import os
import time
import gymnasium as gym

# 导入你的本地魔改版环境
from gymnasium_env.envs.ant_env_v5 import AntEnv

# 导入我们的图网络 Wrapper 和造物主 XML 生成器
from ant_env_test.graph_obs_wrapper import AntGraphObsWrapper, AntActionWrapper
from ant_env_test.co_evolution_main import build_mutant_ant_xml 

def test_mutant_graph_pipeline(num_legs=5):
    print(f"\n🧬 正在培养 {num_legs} 足异形测试体...")
    
    # 1. 动态生成物理世界图纸
    TEMP_XML_DIR = "./temp_xmls"
    os.makedirs(TEMP_XML_DIR, exist_ok=True)
    xml_path = os.path.join(TEMP_XML_DIR, f"test_ant_{num_legs}legs.xml")
    
    xml_str = build_mutant_ant_xml(num_legs=num_legs, thigh_len=0.2, calf_len=0.2)
    with open(xml_path, "w") as f:
        f.write(xml_str)
        
    # 2. 实例化物理容器
    env = AntEnv(xml_file=xml_path, render_mode="human")
    env = gym.wrappers.TimeLimit(env, max_episode_steps=500)
    
    # 3. 🕸️ 植入神经网络接头：套上我们的图网络解析器
    # 这里我们设定宇宙最大容量为 8 条腿，测试 Wrapper 是否能正确 Mask 掉不存在的 3 条腿
    env = AntGraphObsWrapper(env, num_legs=num_legs, max_legs=8)
    env = AntActionWrapper(env, num_legs=num_legs, max_legs=8)
    
    # 4. 唤醒世界
    obs, info = env.reset()
    
    # --------------------------------------------------
    # 📊 维度核对大检阅！(这是 GCN 能够完美运转的生命线)
    # --------------------------------------------------
    print("\n" + "="*50)
    print("✅ 异形环境 + Wrapper 组装成功！GCN 视角的接口维度如下：")
    print(f"  🧠 [Nodes Shape]:  {obs['nodes'].shape}  (预期: 17x4 -> 1躯干+16关节，每个节点4个物理特征)")
    print(f"  🌍 [Global Shape]: {obs['global'].shape}     (预期: (6,) -> 躯干的前庭觉特征)")
    print(f"  💪 [Action Space]: {env.action_space.shape}    (预期: ({num_legs * 2},) -> {num_legs}条腿对应的真实马达数)")
    
    # 打印 Mask，直观确认是否正确屏蔽了未发育出来的腿 (应该前 11 个是 1，后面是 0)
    print(f"  🎭 [有效 Mask]:    {obs['mask']}")
    print("="*50 + "\n")
    
    time.sleep(2) 
    
    # 5. 暴力随机驾驶测试
    print("🌪️ 开启随机动作狂暴测试，观察触地检测和物理张量...")
    total_reward = 0.0
    
    for step in range(500):
        random_action = env.action_space.sample() 
        obs, reward, terminated, truncated, info = env.step(random_action)
        total_reward += reward
        
        # 实时监控节点 1 (第一条腿的 Hip 关节) 的触地状态 (特征索引 2)
        # 如果画面里腿踩在地上，这个值应该是 1.0
        contact_status = obs['nodes'][1][2] 
        
        print(f"\r   [Step {step}] 腿部0触地状态: {contact_status} | 躯干高度: {obs['global'][0]:.3f}", end="")
        
        time.sleep(0.02) 
        
        if terminated or truncated:
            reason = "💀 失去平衡" if terminated else "⏱️ 寿命耗尽"
            print(f"\n   异形停止活动。原因: {reason} | 累计得分: {total_reward:.2f}")
            obs, info = env.reset()
            total_reward = 0.0
            print("⚡ 异形重生...")

    env.close()
    print("\n🎉 全链路干跑测试圆满结束！")

if __name__ == "__main__":
    test_mutant_graph_pipeline(num_legs=5)