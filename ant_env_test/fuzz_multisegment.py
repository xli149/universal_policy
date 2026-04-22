import os
import random
import time
import numpy as np

# 导入你本地的纯净版物理环境 (不需要 GCN Wrapper，我们只测物理会不会崩)
from gymnasium_env.envs.ant_env_v5 import AntEnv

def build_centipede_xml(num_segments):
    """
    🐛 进化版赛博蜈蚣：形状混接 + 截段长度随机抽样
    """
    s = [
        '<mujoco model="centipede">\n',
        '  <compiler angle="degree" coordinate="local" inertiafromgeom="true"/>\n',
        '  <option integrator="RK4" timestep="0.01"/>\n',
        '  <default>\n',
        '    <joint armature="0.1" damping="1" limited="true"/>\n',
        '    <geom conaffinity="1" condim="3" density="5.0" friction="1 0.5 0.5" margin="0.01"/>\n',
        '  </default>\n',
        '  <asset>\n',
        '    <texture builtin="checker" height="512" name="texplane" rgb1=".2 .3 .4" rgb2=".1 0.15 0.2" type="2d" width="512"/>\n',
        '    <material name="MatPlane" reflectance="0.5" texrepeat="60 60" texture="texplane"/>\n',
        '  </asset>\n',
        '  <worldbody>\n',
        '    <light cutoff="100" diffuse="1 1 1" dir="-0 0 -1.3" directional="true" pos="0 0 1.3"/>\n',
        '    <geom material="MatPlane" name="floor" pos="0 0 0" size="40 40 40" type="plane"/>\n'
    ]

    body_xml = ""
    closing_tags = ""
    motor_xml = ['  <actuator>\n']
    
    # 我们用这个变量记录“上一节”抽样出来的长度，用于定位“这一节”的起始位置
    prev_seg_len = 0.25 

    for i in range(num_segments):
        # 🎲 1. 抽样生成当前节的各项参数
        thigh_len = random.uniform(0.1, 0.25)
        calf_len = random.uniform(0.1, 0.25)
        geom_type = random.choice(["capsule", "sphere"])
        
        # 🎲 2. 抽样生成当前截段的物理长度 (不再固定 0.25)
        current_seg_len = random.uniform(0.15, 0.4) 
        
        if i == 0:
            # 第一节（头部）
            body_xml += f'    <body name="seg_{i}" pos="0 0 0.5">\n'
            body_xml += '      <freejoint name="root"/>\n'
        else:
            # 后续体节：偏移距离由【上一节】的长度决定
            body_xml += f'      <body name="seg_{i}" pos="-{prev_seg_len} 0 0">\n'
            body_xml += f'        <joint name="spine_yaw_{i}" type="hinge" axis="0 0 1" range="-40 40"/>\n'
            body_xml += f'        <joint name="spine_pitch_{i}" type="hinge" axis="0 1 0" range="-20 20"/>\n'
            motor_xml.append(f'    <motor joint="spine_yaw_{i}" ctrlrange="-1.0 1.0" gear="30"/>\n')
            motor_xml.append(f'    <motor joint="spine_pitch_{i}" ctrlrange="-1.0 1.0" gear="30"/>\n')

        # 🛠️ 3. 根据抽样形状生成躯干
        if geom_type == "capsule":
            # 胶囊体：长度完美匹配抽样值
            body_xml += f'      <geom name="geom_seg_{i}" type="capsule" fromto="0 0 0 -{current_seg_len:.3f} 0 0" size="0.06" rgba="0.2 0.8 0.4 1"/>\n'
        else:
            # 球体：为了不留空隙，画一根匹配抽样长度的脊椎骨，再把球串中间
            body_xml += f'      <geom name="spine_bone_{i}" type="capsule" fromto="0 0 0 -{current_seg_len:.3f} 0 0" size="0.02" rgba="0.5 0.5 0.5 1"/>\n'
            body_xml += f'      <geom name="geom_seg_{i}" type="sphere" pos="-{current_seg_len/2:.3f} 0 0" size="0.08" rgba="0.4 0.2 0.8 1"/>\n'

        # 🛠️ 4. 为当前体节生成腿 (位置定在当前截段的正中)
        leg_pos_x = -current_seg_len / 2
        for side, angle in [("L", 90), ("R", -90)]:
            body_xml += f"""
              <body name="leg_{i}_{side}" pos="{leg_pos_x:.3f} 0 0" euler="0 0 {angle}">
                <joint name="hip_{i}_{side}" type="hinge" axis="0 0 1" range="-40 40"/>
                <geom name="thigh_geom_{i}_{side}" type="capsule" fromto="0 0 0 {thigh_len:.3f} 0 0" size="0.04" rgba="0.8 0.4 0.2 1"/>
                <body name="ankle_{i}_{side}" pos="{thigh_len:.3f} 0 0">
                  <joint name="knee_{i}_{side}" type="hinge" axis="0 1 0" range="10 70"/>
                  <geom name="calf_geom_{i}_{side}" type="capsule" fromto="0 0 0 {calf_len:.3f} 0 -0.1" size="0.03" rgba="0.8 0.4 0.2 1"/>
                </body>
              </body>
            """
            motor_xml.append(f'    <motor joint="hip_{i}_{side}" ctrlrange="-1.0 1.0" gear="30"/>\n')
            motor_xml.append(f'    <motor joint="knee_{i}_{side}" ctrlrange="-1.0 1.0" gear="30"/>\n')
            
        closing_tags += "    </body>\n"
        
        # 重要：更新偏移量，传给下一次循环使用
        prev_seg_len = current_seg_len

    # 拼装
    s.append(body_xml)
    s.append(closing_tags)
    s.append('  </worldbody>\n')
    motor_xml.append('  </actuator>\n</mujoco>\n')
    s.extend(motor_xml)
    
    return "".join(s)

def fuzz_multisegment_physics(test_rounds=500):
    print(f"🌪️ [阶段一] 开始多段脊椎生物 Fuzzing 物理极限盲测 ({test_rounds} 轮)...\n")
    
    TEMP_XML_DIR = "./temp_fuzz_xmls"
    os.makedirs(TEMP_XML_DIR, exist_ok=True)
    
    survivors = []
    
    for i in range(test_rounds):
        # 随机生成 2 到 6 个体节的蜈蚣 (4腿 到 12腿)
        num_segments = random.randint(2, 6)
        xml_path = os.path.join(TEMP_XML_DIR, f"centipede_{i}.xml")
        xml_str = build_centipede_xml(num_segments)
        
        with open(xml_path, "w") as f:
            f.write(xml_str)
            
        try:
            # 极速加载测试
            env = AntEnv(xml_file=xml_path)
            obs, _ = env.reset()
            
            # 极限扭力暴走测试
            for _ in range(30):
                action = env.action_space.sample() 
                obs, reward, terminated, truncated, info = env.step(action)
                
                # 监控 NaN 爆炸
                if np.isnan(obs['qpos']).any():
                    raise ValueError("脊椎打结引发了物理张量爆炸 (NaN)！")
                if terminated: break
                    
            env.close()
            survivors.append(num_segments)
            
        except Exception as e:
            print(f"\n🚨 崩溃报告: {num_segments} 体节生物发生灾难 -> {e}")
        finally:
            if os.path.exists(xml_path):
                os.remove(xml_path)

    print(f"\n✅ 盲测结束！存活率: {len(survivors)} / {test_rounds}")
    
    if not survivors:
        print("💀 脊椎约束太弱，全部自我毁灭了。需要调小 spine 的 range。")
        return

    # ==========================================
    # 📺 阶段二：抽样可视化
    # ==========================================
    print("\n🎬 [阶段二] 抽取 5 只不同长度的存活机械蜈蚣进行可视化...")
    time.sleep(1)
    
    for i in range(5):
        num_seg = random.choice(survivors)
        print(f"   🐛 正在展示: {num_seg} 体节的机械蜈蚣 (包含 {num_seg*2} 条腿)")
        
        xml_path = os.path.join(TEMP_XML_DIR, "vis_centipede.xml")
        with open(xml_path, "w") as f:
            f.write(build_centipede_xml(num_seg))
            
        env = AntEnv(xml_file=xml_path, render_mode="human")
        env.reset()
        
        for _ in range(200): # 渲染几秒钟
            action = env.action_space.sample()
            env.step(action)
            time.sleep(0.02)
            
        env.close()
        os.remove(xml_path)

if __name__ == "__main__":
    fuzz_multisegment_physics()