from typing import Sequence, Tuple, List
import os
import random

def xml_header(model_name):
    return f'<mujoco model="{model_name}">\n'

def xml_compiler(angle="radian", inertiafromgeom: bool = True):
    return f'  <compiler angle="{angle}" inertiafromgeom="{"true" if inertiafromgeom else "false"}"/>\n'

def xml_default(joint_armature=1, joint_damping=1, joint_limited=True, geom_friction=(1, 0.1, 0.1), geom_rgba=(0.7,0.7,0,1), geom_density=1000):
    fr = " ".join(map(str, geom_friction))
    rgba = " ".join(map(str, geom_rgba))
    return(
         "  <default>\n"
        f'    <joint armature="{joint_armature}" damping="{joint_damping}" limited="{"true" if joint_limited else "false"}"/>\n'
        f'    <geom contype="1" conaffinity="1" friction="{fr}" rgba="{rgba}" density="{geom_density}"/>\n'
        "  </default>\n"
    )

# 🚀 核心修改 1：降低 timestep 到 0.005，让物理引擎的“视力”变好，防止高速穿模
def xml_option(gravity=(0,0,-9.81), integrator="RK4", timestep=0.005) -> str:
    g = " ".join(map(str, gravity))
    return f'  <option gravity="{g}" integrator="{integrator}" timestep="{timestep}"/>\n'

def xml_arena(arena_half: float = 0.45, wall_radius: float = 0.02) -> str:
    # ground plane
    s = []
    s.append('  <worldbody>\n')
    s.append(
        f'    <geom conaffinity="0" contype="0" name="ground" pos="0 0 0" '
        f'rgba="0.9 0.9 0.9 1" size="{arena_half} {arena_half} 10" type="plane"/>\n'
    )
    # 4 boundary capsules
    s.append(f'    <geom conaffinity="0" contype="0" name="sideS" rgba="0.9 0.4 0.6 1" size="{wall_radius}" type="capsule" fromto="{-arena_half} {-arena_half} .01 {arena_half} {-arena_half} .01"/>\n')
    s.append(f'    <geom conaffinity="0" contype="0" name="sideE" rgba="0.9 0.4 0.6 1" size="{wall_radius}" type="capsule" fromto="{arena_half} {-arena_half} .01 {arena_half} {arena_half} .01"/>\n')
    s.append(f'    <geom conaffinity="0" contype="0" name="sideN" rgba="0.9 0.4 0.6 1" size="{wall_radius}" type="capsule" fromto="{-arena_half} {arena_half} .01 {arena_half} {arena_half} .01"/>\n')
    s.append(f'    <geom conaffinity="0" contype="0" name="sideW" rgba="0.9 0.4 0.6 1" size="{wall_radius}" type="capsule" fromto="{-arena_half} {-arena_half} .01 {-arena_half} {arena_half} .01"/>\n')
    # root marker
    s.append('    <geom conaffinity="0" contype="0" name="root" rgba="0.9 0.4 0.6 1" size=".011" type="cylinder" fromto="0 0 0 0 0 0.02"/>\n')
    return "".join(s)

def xml_arm_chain(
    lengths: Sequence[float],
    link_radius: float = 0.01,
    fingertip_radius: float = 0.01,
    joint_sphere_radius: float = 0.02, 
    joint_axis=(0, 0, 1),
    joint_range=(-1.5, 1.5),
    joint0_limited: bool = False,
    base_pos=(0, 0, 0.01),
) -> Tuple[str, List[str]]:
    axis = " ".join(map(str, joint_axis))
    jmin, jmax = joint_range
    bx, by, bz = base_pos

    joint_names: List[str] = []
    s: List[str] = []

    # open body0
    s.append(f'    <body name="body0" pos="{bx} {by} {bz}">\n')
    opened_bodies = 1

    for i, L in enumerate(lengths):
        # 🚀 核心修改 2：给连杆加上 margin="0.005"，穿上一层无形的排斥力场护盾
        s.append(
            f'      <geom name="link{i}" type="capsule" size="{link_radius}" '
            f'rgba="0.0 0.4 0.6 1" fromto="0 0 0 {L} 0 0" margin="0.005"/>\n'
        )

        # 2. joint
        jname = f"joint{i}"
        joint_names.append(jname)

        if i == 0:
            limited = "true" if joint0_limited else "false"
            s.append(f'      <joint name="{jname}" type="hinge" pos="0 0 0" axis="{axis}" limited="{limited}"/>\n')
        else:
            s.append(
                f'      <joint name="{jname}" type="hinge" pos="0 0 0" axis="{axis}" limited="true" range="{jmin} {jmax}"/>\n'
            )
        
        # 调试用：鲜红色的防穿模球
        s.append(
            f'      <geom name="joint_guard_{i}" type="sphere" size="{joint_sphere_radius}" '
            f'rgba="1 0 0 1" pos="0 0 0" contype="1" conaffinity="1" margin="0.002"/>\n'
        )

        # 4. open next body or fingertip
        is_last = (i == len(lengths) - 1)
        if not is_last:
            s.append(f'      <body name="body{i+1}" pos="{L} 0 0">\n')
            opened_bodies += 1
        else:
            s.append(f'      <body name="fingertip" pos="{L} 0 0">\n')
            s.append(
                f'        <geom name="fingertip_col" type="sphere" size="{max(fingertip_radius, 0.02)}" '
                f'rgba="0 0 0 0" pos="0 0 0" contype="1" conaffinity="1" margin="0.002"/>\n'
            )
            s.append(
                f'        <geom name="fingertip_vis" type="sphere" size="{fingertip_radius}" '
                f'rgba="0.0 0.8 0.6 1" pos="0 0 0" contype="0" conaffinity="0"/>\n'
            )
            s.append('      </body>\n')

    # close all opened bodies (body0..bodyN-1)
    for _ in range(opened_bodies):
        s.append('    </body>\n')

    return "".join(s), joint_names

def xml_target(
    target_init=(0.1, -0.1, 0.01),
    target_range=0.405,
    target_radius=0.02,
) -> str:
    x, y, z = target_init
    r = target_range
    return (
        f'    <body name="target" pos="{x} {y} {z}">\n'
        f'      <joint name="target_x" type="slide" axis="1 0 0" limited="true" range="{-r} {r}" ref="{x}" damping="0" armature="0" stiffness="0"/>\n'
        f'      <joint name="target_y" type="slide" axis="0 1 0" limited="true" range="{-r} {r}" ref="{y}" damping="0" armature="0" stiffness="0"/>\n'
        f'      <geom name="target" type="sphere" size="{target_radius}" rgba="0.9 0.2 0.2 1" pos="0 0 0" contype="0" conaffinity="0"/>\n'
        f'    </body>\n'
    )

def xml_actuators(
    joint_names: Sequence[str],
    ctrlrange=(-1.0, 1.0),
    gear=100.0,
) -> str:
    c0, c1 = ctrlrange
    s = []
    s.append("  <actuator>\n")
    for j in joint_names:
        s.append(f'    <motor joint="{j}" ctrllimited="true" ctrlrange="{c0} {c1}" gear="{gear}"/>\n')
    s.append("  </actuator>\n")
    return "".join(s)

def build_reacher_xml(
    model_name: str,
    lengths: Sequence[float],
    link_radius: float = 0.01,
    fingertip_radius: float = 0.01,
    joint_range=(-2.5, 2.5),
    gear=100.0,
    ctrlrange=(-1.0, 1.0),
    arena_margin: float = 1.3,
    target_ratio: float = 0.9,
) -> str:
    R, arena_half, target_range = reacher_scale(lengths, arena_margin=arena_margin, target_ratio=target_ratio)

    pieces: List[str] = []
    pieces.append(xml_header(model_name))
    pieces.append(xml_compiler())
    pieces.append(xml_default())
    pieces.append(xml_option())

    pieces.append(xml_arena(arena_half=arena_half))

    arm_xml, joint_names = xml_arm_chain(
        lengths=lengths,
        link_radius=link_radius,
        fingertip_radius=fingertip_radius,
        joint_range=joint_range,
    )
    pieces.append(arm_xml)

    pieces.append(xml_target(target_range=target_range))

    pieces.append("  </worldbody>\n")
    pieces.append(xml_actuators(joint_names, ctrlrange=ctrlrange, gear=gear))
    pieces.append("</mujoco>\n")

    return "".join(pieces)

def reacher_scale(lengths: Sequence[float], arena_margin: float = 1.3, target_ratio: float = 0.9):
    R = float(sum(lengths))
    arena_half = R * arena_margin
    target_range = R * target_ratio
    return R, arena_half, target_range

if __name__ == "__main__":
    # 1. 创建专门存放通用环境池的文件夹
    output_dir = "./gymnasium_env/envs/universal_pool"
    os.makedirs(output_dir, exist_ok=True)

    # 2. 定义变异的知识结界 (网络未来能学会插值的锚点)
    LENGTH_OPTIONS = [0.05, 0.10, 0.15]
    
    # 挑战 1 到 10 关节！
    JOINT_COUNTS = range(1, 11) 
    
    # 每个关节数随机抽样的变异体数量
    SAMPLES_PER_JOINT = 20  

    total_generated = 0
    random.seed(42)  # 固定种子保证可复现

    print("🌌 开始生成 1-10 关节的大千世界 XML...")
    
    for num_joints in JOINT_COUNTS:
        unique_combinations = set()
        
        # 对于1关节只有3种可能，防止死循环
        max_possible = len(LENGTH_OPTIONS) ** num_joints
        target_samples = min(SAMPLES_PER_JOINT, max_possible)
        
        while len(unique_combinations) < target_samples:
            combo = tuple(random.choices(LENGTH_OPTIONS, k=num_joints))
            unique_combinations.add(combo)
            
        for lengths in unique_combinations:
            lengths_str = "_".join([f"{l:.2f}" for l in lengths])
            model_name = f"reacher_{num_joints}j_{lengths_str}"
            file_path = os.path.join(output_dir, f"{model_name}.xml")
            
            xml_str = build_reacher_xml(
                model_name=model_name,
                lengths=lengths,
                link_radius=0.015,
                joint_range=(-2.2, 2.2), # 🚀 核心修改 3：收紧角度，防止过度折叠
                gear=40.0,               # 🚀 核心修改 4：稍微降低马力，防止速度过快引发隧穿
                arena_margin=1.35,  
                target_ratio=0.9,
            )
            
            with open(file_path, "w") as f:
                f.write(xml_str)
            
            total_generated += 1
            
        print(f"✅ {num_joints:02d} 关节组: 成功生成 {len(unique_combinations):02d} 个变异形态")

    print("="*40)
    print(f"🎉 大功告成！成功生成 {total_generated} 个绝不穿模的 XML 环境！")
    print(f"它们被统一保存在: {output_dir}")