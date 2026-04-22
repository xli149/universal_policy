import os
import time
import random
import mujoco
import mujoco.viewer

# ==========================================
# 1. 终极造物主引擎
# ==========================================
def build_mutant_ant_xml(genome):
    num_legs = len(genome)
    max_leg_reach = 0.0
    for leg_gene in genome:
        total_len = leg_gene[1] + leg_gene[2]
        if total_len > max_leg_reach:
            max_leg_reach = total_len
            
    spawn_z = max_leg_reach + 0.2
    
    s = [
        '<mujoco model="mutant_ant">\n',
        '  <compiler angle="degree" coordinate="local" inertiafromgeom="true"/>\n',
        '  <option integrator="RK4" timestep="0.01"/>\n',
        '  <default>\n',
        '    <joint armature="1" damping="1" limited="true"/>\n',
        '    <geom conaffinity="1" condim="3" density="5.0" friction="1 0.5 0.5" margin="0.01" rgba="0.8 0.6 0.4 1"/>\n',
        '  </default>\n',
        '  <asset>\n',
        '    <texture builtin="checker" height="512" name="texplane" rgb1=".2 .3 .4" rgb2=".1 0.15 0.2" type="2d" width="512"/>\n',
        '    <material name="MatPlane" reflectance="0.5" shininess="1" specular="1" texrepeat="60 60" texture="texplane"/>\n',
        '  </asset>\n',
        '  <worldbody>\n',
        '    <light cutoff="100" diffuse="1 1 1" dir="-0 0 -1.3" directional="true" exponent="1" pos="0 0 1.3" specular=".1 .1 .1"/>\n',
        '    <geom conaffinity="1" condim="3" material="MatPlane" name="floor" pos="0 0 0" rgba="0.8 0.9 0.8 1" size="40 40 40" type="plane"/>\n',
        
        f'    <body name="torso" pos="0 0 {spawn_z:.3f}">\n',
        
        '      <camera name="track" mode="trackcom" pos="0 -3 2.0" euler="60 0 0"/>\n',
        '      <freejoint name="root"/>\n',
        '      <geom name="torso_geom" pos="0 0 0" size="0.15" type="sphere"/>\n'
    ]
    
    for i, leg_gene in enumerate(genome):
        angle, thigh_len, calf_len, hip_min, hip_max, ankle_min, ankle_max = leg_gene
        hip_min, hip_max = min(hip_min, hip_max), max(hip_min, hip_max)
        ankle_min, ankle_max = min(ankle_min, ankle_max), max(ankle_min, ankle_max)
        
        s.extend([
            f'      <body name="leg_{i}" pos="0 0 0" euler="0 0 {angle:.3f}">\n',
            f'        <geom fromto="0.0 0.0 0.0 0.1 0.0 0.0" name="aux_{i}_geom" size="0.04" type="capsule"/>\n',
            f'        <body name="leg_{i}_aux" pos="0.1 0.0 0">\n',
            f'          <joint axis="0 0 1" name="hip_{i}" pos="0.0 0.0 0.0" range="{hip_min:.3f} {hip_max:.3f}" type="hinge"/>\n',
            f'          <geom fromto="0.0 0.0 0.0 {thigh_len:.3f} 0.0 0.0" name="thigh_{i}_geom" size="0.04" type="capsule"/>\n',
            f'          <body pos="{thigh_len:.3f} 0.0 0" name="leg_{i}_ankle">\n',
            f'            <joint axis="0 1 0" name="ankle_{i}" pos="0.0 0.0 0.0" range="{ankle_min:.3f} {ankle_max:.3f}" type="hinge"/>\n',
            f'            <geom fromto="0.0 0.0 0.0 {calf_len:.3f} 0.0 -0.1" name="calf_{i}_geom" size="0.04" type="capsule"/>\n',
            '          </body>\n',
            '        </body>\n',
            '      </body>\n'
        ])
    
    s.extend(['    </body>\n  </worldbody>\n  <actuator>\n'])
    
    for i in range(num_legs):
        s.append(f'    <motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="hip_{i}" gear="30"/>\n')
        s.append(f'    <motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="ankle_{i}" gear="30"/>\n')
        
    s.append('  </actuator>\n</mujoco>\n')
    return "".join(s)

# ==========================================
# 2. 辐射对称基因生成器 (支持奇/偶数腿)
# ==========================================
def generate_random_genome(target_legs=None, min_leg=4, max_leg=8):
    """
    如果传入了 target_legs，就生成指定数量的腿；
    否则就在 4~8 之间随机生成。
    """
    num_legs = target_legs if target_legs is not None else random.randint(min_leg, max_leg)
    genome = []

    for i in range(num_legs):
        # 🌟 核心：辐射对称的基准角度
        base_angle = (360 / num_legs) * i
        
        # 允许微小的基因突变偏差 (正负5度以内，保证不破坏整体重心)
        noise = random.uniform(-5, 5) 
        angle = base_angle + noise

        # 设定一个基准长度，然后在这个基准上轻微波动，防止初代长短腿太夸张站不住
        base_thigh = random.uniform(0.15, 0.3)
        base_calf = random.uniform(0.15, 0.3)
        
        thigh_len = base_thigh * random.uniform(0.9, 1.1)
        calf_len = base_calf * random.uniform(0.9, 1.1)

        hip_min = random.uniform(-35, -25)
        hip_max = random.uniform(25, 35)
        ankle_min = random.uniform(20, 40)
        ankle_max = random.uniform(60, 80)

        leg_gene = [angle, thigh_len, calf_len, hip_min, hip_max, ankle_min, ankle_max]
        genome.append(leg_gene)

    return genome

# ==========================================
# 3. 试管测试主程序
# ==========================================
if __name__ == "__main__":
    print("🧪 准备进行辐射对称物理测试...")
    
    # 🎯 核心测试开关：你想测试几条腿的生物？
    # 试试改成 4, 5, 6, 7, 8！或者设为 None 进行盲盒抽卡。
    TARGET_LEGS = 7
    
    # 1. 召唤基因生成器
    test_genome = generate_random_genome(target_legs=TARGET_LEGS)
    print(f"🧬 成功生成一只 {len(test_genome)} 足异形蚂蚁的基因！")

    # 2. 召唤 XML 渲染器
    xml_content = build_mutant_ant_xml(test_genome)

    # 3. 保存 XML 文件
    os.makedirs("./temp_xmls", exist_ok=True)
    test_xml_path = "./temp_xmls/test_tube_ant.xml"
    with open(test_xml_path, "w") as f:
        f.write(xml_content)
    
    print(f"✅ XML 已成功保存至: {test_xml_path}")

    # 4. 原生 MuJoCo 渲染大舞台！
    try:
        print("🚀 正在启动 MuJoCo 渲染器...")
        print("   👉 操作提示：双击选中躯干，按住鼠标【右键】可拖拽其在空中的位置！")
        
        model = mujoco.MjModel.from_xml_string(xml_content)
        data = mujoco.MjData(model)
        
        mujoco.viewer.launch(model, data)
        
    except Exception as e:
        print(f"\n❌ MuJoCo 解析失败！错误信息：\n{e}")