import time
# 从你的本地路径导入原版环境
from gymnasium_env.envs.ant_env_v5 import AntEnv

def test_pure_local_ant():
    print("🐜 正在启动本地(已字典化魔改) Ant-v5 基线测试...")
    
    # 替换为你本机的实际路径
    env = AntEnv(xml_file="/Users/chrislee/Documents/mujoco_test/gymnasium_env/envs/ant_env.xml", render_mode="human")
    
    obs, info = env.reset()
    print("\n✅ 环境加载成功！底层输出已成功进化为物理字典：")
    
    # 🚀 核心修复：遍历字典，打印每一个物理组件的维度
    for key, val in obs.items():
        print(f"  👁️ [obs['{key}'] 维度]: {val.shape}")
        
    print(f"  💪 [Action 维度]:        {env.action_space.shape} (标准 4 足蚂蚁预期为: (8,))")
    
    print("\n🌪️ 开始随机动作干跑...")
    for step in range(500):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        time.sleep(0.02)
        
        if terminated or truncated:
            print(f"   [Step {step}] 蚂蚁翻车，重新生成...")
            env.reset()
            
    env.close()
    print("🎉 本地测试跑通！你的环境已经 100% 准备好接入 GCN Wrapper 了！")

if __name__ == "__main__":
    test_pure_local_ant()