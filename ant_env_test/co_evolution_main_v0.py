import math

def build_mutant_ant_xml(num_legs, thigh_len=0.2, calf_len=0.2):
    """
    🧬 初级造物主：动态辐射蚂蚁 XML 生成器 (对称形态)
    根据输入的腿数，在躯干周围 360 度均匀生成多段式腿部。
    """
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
        '    <body name="torso" pos="0 0 0.75">\n',
        '     <camera name="track" mode="trackcom" pos="0 -3 2.0" euler="60 0 0"/>\n',
        '      <freejoint name="root"/>\n',
        '      <geom name="torso_geom" pos="0 0 0" size="0.15" type="sphere"/>\n'
    ]
    
    for i in range(num_legs):
        angle = (360 / num_legs) * i
        
        s.extend([
            f'      <body name="leg_{i}" pos="0 0 0" euler="0 0 {angle}">\n',
            f'        <geom fromto="0.0 0.0 0.0 0.1 0.0 0.0" name="aux_{i}_geom" size="0.04" type="capsule"/>\n',
            f'        <body name="leg_{i}_aux" pos="0.1 0.0 0">\n',
            f'          <joint axis="0 0 1" name="hip_{i}" pos="0.0 0.0 0.0" range="-30 30" type="hinge"/>\n',
            f'          <geom fromto="0.0 0.0 0.0 {thigh_len} 0.0 0.0" name="thigh_{i}_geom" size="0.04" type="capsule"/>\n',
            f'          <body pos="{thigh_len} 0.0 0" name="leg_{i}_ankle">\n',
            f'            <joint axis="0 1 0" name="ankle_{i}" pos="0.0 0.0 0.0" range="30 70" type="hinge"/>\n',
            f'            <geom fromto="0.0 0.0 0.0 {calf_len} 0.0 -0.1" name="calf_{i}_geom" size="0.04" type="capsule"/>\n',
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

def build_asymmetric_ant_xml(genome):
    """
    🧬 终极造物主：非对称异形 XML 生成器
    接收一个高维基因矩阵，支持每一条腿的独立突变！
    genome = [
        [附着角度, 大腿长, 小腿长, 髋关节范围min, 髋关节范围max, 踝关节范围min, 踝关节范围max],
        ...
    ]
    """
    num_legs = len(genome)
    
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
        '    <body name="torso" pos="0 0 0.75">\n',
        '      <camera name="track" mode="trackcom" pos="0 -3 2.0" euler="60 0 0"/>\n',
        '      <freejoint name="root"/>\n',
        '      <geom name="torso_geom" pos="0 0 0" size="0.15" type="sphere"/>\n'
    ]
    
    # 🧬 遍历基因矩阵，生成非对称肢体
    for i, leg_gene in enumerate(genome):
        angle, thigh_len, calf_len, hip_min, hip_max, ankle_min, ankle_max = leg_gene
        
        # 🛡️ 防爆机制：确保关节范围合法 (min 必须小于 max)，防止 MuJoCo 报错
        hip_min, hip_max = min(hip_min, hip_max), max(hip_min, hip_max)
        ankle_min, ankle_max = min(ankle_min, ankle_max), max(ankle_min, ankle_max)
        
        # ⚠️ 注意这里加了 {:.3f} 限制小数点，因为 MuJoCo 解析超长浮点数时偶尔会抽风
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