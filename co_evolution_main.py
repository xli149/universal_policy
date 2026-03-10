import os
import random
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import TimeLimit
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from typing import Sequence, Tuple, List
import math
# 导入你自定义的 Wrapper
import gymnasium_env
from graph_obs_wrapper import PaddedGraphObsWrapper, PaddedActionWrapper
import time # 如果开头没 import
# ==========================================
# ⚙️ 全局配置
# ==========================================
ENV_ID = "gymnasium_env/Reacher2D-v0"
MAX_JOINTS = 10
MAX_EPISODE_STEPS = 100

# 🚀 指向你的神级基础大脑
PRETRAINED_MODEL_PATH = "./checkpoints/pretrained_base_brain_v1/pretrained_base_brain"

# 临时存放动态生成的 XML 的目录
TEMP_XML_DIR = "./temp_xmls"
os.makedirs(TEMP_XML_DIR, exist_ok=True)

# ==========================================
# 🧬 第一部分：模块化造物主工厂 (基于你的严谨物理引擎)
# ==========================================
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

def xml_option(gravity=(0,0,-9.81), integrator="RK4", timestep=0.005) -> str:
    g = " ".join(map(str, gravity))
    return f'  <option gravity="{g}" integrator="{integrator}" timestep="{timestep}"/>\n'

def xml_arena(arena_half: float = 0.45, wall_radius: float = 0.02) -> str:
    s = []
    s.append('  <worldbody>\n')
    s.append(
        f'    <geom conaffinity="0" contype="0" name="ground" pos="0 0 0" '
        f'rgba="0.9 0.9 0.9 1" size="{arena_half} {arena_half} 10" type="plane"/>\n'
    )
    s.append(f'    <geom conaffinity="0" contype="0" name="sideS" rgba="0.9 0.4 0.6 1" size="{wall_radius}" type="capsule" fromto="{-arena_half} {-arena_half} .01 {arena_half} {-arena_half} .01"/>\n')
    s.append(f'    <geom conaffinity="0" contype="0" name="sideE" rgba="0.9 0.4 0.6 1" size="{wall_radius}" type="capsule" fromto="{arena_half} {-arena_half} .01 {arena_half} {arena_half} .01"/>\n')
    s.append(f'    <geom conaffinity="0" contype="0" name="sideN" rgba="0.9 0.4 0.6 1" size="{wall_radius}" type="capsule" fromto="{-arena_half} {arena_half} .01 {arena_half} {arena_half} .01"/>\n')
    s.append(f'    <geom conaffinity="0" contype="0" name="sideW" rgba="0.9 0.4 0.6 1" size="{wall_radius}" type="capsule" fromto="{-arena_half} {-arena_half} .01 {-arena_half} {arena_half} .01"/>\n')
    s.append('    <geom conaffinity="0" contype="0" name="root" rgba="0.9 0.4 0.6 1" size=".011" type="cylinder" fromto="0 0 0 0 0 0.02"/>\n')
    return "".join(s)

def xml_arm_chain(
    lengths: Sequence[float], link_radius: float = 0.01, fingertip_radius: float = 0.01,
    joint_sphere_radius: float = 0.02, joint_axis=(0, 0, 1), joint_range=(-1.5, 1.5),
    joint0_limited: bool = False, base_pos=(0, 0, 0.01),
) -> Tuple[str, List[str]]:
    axis = " ".join(map(str, joint_axis))
    jmin, jmax = joint_range
    bx, by, bz = base_pos
    joint_names, s = [], []
    s.append(f'    <body name="body0" pos="{bx} {by} {bz}">\n')
    opened_bodies = 1

    for i, L in enumerate(lengths):
        s.append(
            f'      <geom name="link{i}" type="capsule" size="{link_radius}" '
            f'rgba="0.0 0.4 0.6 1" fromto="0 0 0 {L} 0 0" margin="0.005"/>\n'
        )
        jname = f"joint{i}"
        joint_names.append(jname)
        if i == 0:
            limited = "true" if joint0_limited else "false"
            s.append(f'      <joint name="{jname}" type="hinge" pos="0 0 0" axis="{axis}" limited="{limited}"/>\n')
        else:
            s.append(f'      <joint name="{jname}" type="hinge" pos="0 0 0" axis="{axis}" limited="true" range="{jmin} {jmax}"/>\n')
        
        s.append(f'      <geom name="joint_guard_{i}" type="sphere" size="{joint_sphere_radius}" rgba="1 0 0 1" pos="0 0 0" contype="1" conaffinity="1" margin="0.002"/>\n')

        if i != len(lengths) - 1:
            s.append(f'      <body name="body{i+1}" pos="{L} 0 0">\n')
            opened_bodies += 1
        else:
            s.append(f'      <body name="fingertip" pos="{L} 0 0">\n')
            s.append(f'        <geom name="fingertip_col" type="sphere" size="{max(fingertip_radius, 0.02)}" rgba="0 0 0 0" pos="0 0 0" contype="1" conaffinity="1" margin="0.002"/>\n')
            s.append(f'        <geom name="fingertip_vis" type="sphere" size="{fingertip_radius}" rgba="0.0 0.8 0.6 1" pos="0 0 0" contype="0" conaffinity="0"/>\n')
            s.append('      </body>\n')

    for _ in range(opened_bodies):
        s.append('    </body>\n')
    return "".join(s), joint_names



# def xml_target(target_range=0.405, target_radius=0.02) -> str:
#     r = target_range
    
#     # 🚀 终极保险机制：在生成 XML 时，动态计算一个合法的初始坐标！
#     while True:
#         # 在合法的滑轨正方形范围内摇号
#         init_x = random.uniform(-r, r)
#         init_y = random.uniform(-r, r)
        
#         # 1. math.hypot 算出距离原点的直线距离（圆的半径）
#         # 2. 必须 < r：确保它在合法的圆内
#         # 3. 必须 > 0.05：确保它不要和原点那个柱子(root)重叠穿模
#         if 0.05 < math.hypot(init_x, init_y) < r:
#             break
            
#     # 动态将合法坐标注入到 pos 和 ref 中
#     return (
#         f'    <body name="target" pos="{init_x:.4f} {init_y:.4f} 0.01">\n'
#         f'      <joint name="target_x" type="slide" axis="1 0 0" limited="true" range="{-r} {r}" ref="{init_x:.4f}" damping="0" armature="0" stiffness="0"/>\n'
#         f'      <joint name="target_y" type="slide" axis="0 1 0" limited="true" range="{-r} {r}" ref="{init_y:.4f}" damping="0" armature="0" stiffness="0"/>\n'
#         f'      <geom name="target" type="sphere" size="{target_radius}" rgba="0.9 0.2 0.2 1" pos="0 0 0" contype="0" conaffinity="0"/>\n'
#         f'    </body>\n'
#     )
# import math
# import random

def xml_target(target_range=0.405, target_radius=0.02) -> str:
    r = target_range
    
    # 🚀 修复死循环：动态计算内圈安全距离
    # 既要避开中心柱子(半径0.011)，又要绝对保证比外圈 r 小，留出摇号空间
    inner_bound = min(0.015, r * 0.5) 
    
    while True:
        init_x = random.uniform(-r, r)
        init_y = random.uniform(-r, r)
        
        # 现在的逻辑：必须在动态内圈和外圈之间！
        if inner_bound < math.hypot(init_x, init_y) < r:
            break
            
    return (
        f'    <body name="target" pos="{init_x:.4f} {init_y:.4f} 0.01">\n'
        f'      <joint name="target_x" type="slide" axis="1 0 0" limited="true" range="{-r} {r}" ref="{init_x:.4f}" damping="0" armature="0" stiffness="0"/>\n'
        f'      <joint name="target_y" type="slide" axis="0 1 0" limited="true" range="{-r} {r}" ref="{init_y:.4f}" damping="0" armature="0" stiffness="0"/>\n'
        f'      <geom name="target" type="sphere" size="{target_radius}" rgba="0.9 0.2 0.2 1" pos="0 0 0" contype="0" conaffinity="0"/>\n'
        f'    </body>\n'
    )
def xml_actuators(joint_names: Sequence[str], ctrlrange=(-1.0, 1.0), gear=100.0) -> str:
    c0, c1 = ctrlrange
    s = ["  <actuator>\n"]
    for j in joint_names:
        s.append(f'    <motor joint="{j}" ctrllimited="true" ctrlrange="{c0} {c1}" gear="{gear}"/>\n')
    s.append("  </actuator>\n")
    return "".join(s)

def reacher_scale(lengths: Sequence[float], arena_margin: float = 1.3, target_ratio: float = 0.9):
    R = float(sum(lengths))
    return R, R * arena_margin, R * target_ratio

def build_reacher_xml(model_name: str, lengths: Sequence[float], link_radius: float = 0.01,
                      fingertip_radius: float = 0.01, joint_range=(-2.5, 2.5), gear=100.0,
                      ctrlrange=(-1.0, 1.0), arena_margin: float = 1.3, target_ratio: float = 0.9) -> str:
    R, arena_half, target_range = reacher_scale(lengths, arena_margin=arena_margin, target_ratio=target_ratio)
    pieces = [
        xml_header(model_name), xml_compiler(), xml_default(), xml_option(),
        xml_arena(arena_half=arena_half)
    ]
    arm_xml, joint_names = xml_arm_chain(lengths=lengths, link_radius=link_radius, fingertip_radius=fingertip_radius, joint_range=joint_range)
    pieces.extend([arm_xml, xml_target(target_range=target_range), "  </worldbody>\n", xml_actuators(joint_names, ctrlrange=ctrlrange, gear=gear), "</mujoco>\n"])
    return "".join(pieces)

# ==========================================
# 🧬 第二部分：环境与进化逻辑
# ==========================================
def make_evo_env(lengths, render_mode=None):
    """🧠 将变异的基因转化为物理环境"""
    n_arm_joints = len(lengths)
    lengths_str = "_".join([f"{l:.2f}" for l in lengths])
    model_name = f"evo_{n_arm_joints}j_{lengths_str}"
    xml_filename = os.path.join(TEMP_XML_DIR, f"{model_name}.xml")
    
    # 注入你在独立脚本中完美调试过的物理参数！
    xml_str = build_reacher_xml(
        model_name=model_name,
        lengths=lengths,
        link_radius=0.015,
        joint_range=(-2.2, 2.2), # 防止过度折叠
        gear=40.0,               # 降低马力，防止速度过快引发隧穿
        arena_margin=1.35,  
        target_ratio=0.9,
    )
    
    with open(xml_filename, "w") as f:
        f.write(xml_str)
        
    def _init():
        env = gym.make(ENV_ID, xml_file=xml_filename, render_mode=render_mode, frame_skip=4)
        env = TimeLimit(env, max_episode_steps=MAX_EPISODE_STEPS)
        env = PaddedGraphObsWrapper(env, max_joints=MAX_JOINTS, n_arm_joints=n_arm_joints)
        env = PaddedActionWrapper(env, max_joints=MAX_JOINTS, n_arm_joints=n_arm_joints)
        return env
    return _init

class RobotIndividual:
    def __init__(self, lengths):
        self.lengths = lengths
        self.n_joints = len(lengths)
        self.fitness = -9999.0

def mutate_genome(lengths):
    """基因变异"""
    new_lengths = list(lengths)
    rand_val = random.random()
    
    if rand_val < 0.15 and len(new_lengths) < MAX_JOINTS: 
        new_lengths.append(random.choice([0.05, 0.10, 0.15]))
    elif rand_val < 0.30 and len(new_lengths) > 1:
        new_lengths.pop()
    else:
        idx = random.randint(0, len(new_lengths) - 1)
        new_lengths[idx] = random.choice([0.05, 0.10, 0.15])
    return new_lengths

def calculate_fitness(individual, model, n_episodes=3):
    """零样本打分 + 进化复杂度奖金"""
    venv = DummyVecEnv([make_evo_env(individual.lengths)])
    total_reward = 0.0
    for _ in range(n_episodes):
        obs = venv.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done_arr, _ = venv.step(action)
            total_reward += float(reward[0])
            done = done_arr[0]
            
    venv.close()
    # 复杂度奖金：鼓励长出新关节，对抗动作惩罚
    return (total_reward / n_episodes) + (individual.n_joints * 3.0)

# ==========================================
# 🚀 第三部分：达尔文大循环！
# ==========================================
# ==========================================
# 🚀 第三部分：达尔文大循环！
# ==========================================
if __name__ == "__main__":
    if not os.path.exists(PRETRAINED_MODEL_PATH + ".zip"):
        raise FileNotFoundError(f"🚨 找不到预训练基础大脑: {PRETRAINED_MODEL_PATH}.zip")

    temp_env = DummyVecEnv([make_evo_env([0.10])])
    print(f"🧠 1. 加载第一级火箭：基础大脑 ({PRETRAINED_MODEL_PATH})...")
    gcn_model = SAC.load(PRETRAINED_MODEL_PATH, env=temp_env, device="auto")
    temp_env.close()
    
    # --------------------------------------------------
    # 🧪 测试专用参数 (Dry Run) —— 测试无误后请改回原值！
    # --------------------------------------------------
    POPULATION_SIZE = 30     # 原值: 30
    GENERATIONS = 50        # 原值: 50
    ELITE_K = 5             # 原值: 5
    RL_STEPS_PER_GEN = 15000 # 原值: 15000 
    
    population = [RobotIndividual([random.choice([0.05, 0.10, 0.15]) for _ in range(random.randint(1, 4))]) for _ in range(POPULATION_SIZE)]
    evo_ckpt_dir = "./checkpoints/co_evolution"
    os.makedirs(evo_ckpt_dir, exist_ok=True)
    
    # 🚀 记录仪初始化：记录历史最高分
    global_best_fitness = -9999.0
    
    for gen in range(GENERATIONS):
        print(f"\n{'='*50}\n🧬 达尔文纪元：第 {gen + 1} 代开始\n{'='*50}")
        print("🔍 GCN 大脑正在对生命体进行 Zero-Shot 评估...")
        
        for i, ind in enumerate(population):
            ind.fitness = calculate_fitness(ind, gcn_model)
            if (i+1) % 10 == 0: print(f"   已评估 {i+1}/{POPULATION_SIZE} 个体...")
            
        population.sort(key=lambda x: x.fitness, reverse=True)
        elites = population[:ELITE_K]
        best_individual = elites[0]
        
        print(f"👑 本代最强霸主: {best_individual.n_joints} 关节, 长度={best_individual.lengths}, 适应度={best_individual.fitness:.2f}")
        
        # ==================================================
        # 📸 启动化石记录仪：里程碑 + 破纪录名人堂
        # ==================================================
        lengths_str = "_".join([f"{l:.2f}" for l in best_individual.lengths])
        is_milestone = gen in [0, 9, 19, 29, 39, 49]
        is_new_record = best_individual.fitness > global_best_fitness
        
        if is_milestone or is_new_record:
            milestone_dir = "./checkpoints/evolution_milestones"
            os.makedirs(milestone_dir, exist_ok=True)
            
            if is_new_record:
                global_best_fitness = best_individual.fitness
                prefix = f"RECORD_gen{gen+1}"
            else:
                prefix = f"gen{gen+1}"
                
            specimen_name = f"{prefix}_best_{best_individual.n_joints}j_{lengths_str}"
            
            # 1. 保存躯体图纸 (XML)
            xml_str = build_reacher_xml(
                model_name=specimen_name,
                lengths=best_individual.lengths,
                link_radius=0.015,
                joint_range=(-2.2, 2.2),
                gear=40.0,
                arena_margin=1.35,
                target_ratio=0.9
            )
            with open(os.path.join(milestone_dir, f"{specimen_name}.xml"), "w") as f:
                f.write(xml_str)
                
            # 2. 保存成绩单 (TXT)
            with open(os.path.join(milestone_dir, "evolution_log.txt"), "a") as f:
                f.write(f"=== 🧬 Generation {gen+1} {'(🏆 NEW RECORD!)' if is_new_record else ''} ===\n")
                f.write(f"  👑 最强基因: {best_individual.lengths}\n")
                f.write(f"  ⚙️ 关节数量: {best_individual.n_joints}\n")
                f.write(f"  📈 综合适应度: {best_individual.fitness:.4f}\n\n")
            print(f"   [💾 已存档] 霸主数据已写入化石库: {specimen_name}")
        # ==================================================

        # if gen % 10 == 0:
        #     print(f"📺 正在开启第 {gen+1} 代霸主阅兵模式...")
        #     # 使用 render_mode="human" 创建一个临时环境跑一回合
        #     eval_env_fn = make_evo_env(best_individual.lengths, render_mode="human")
        #     eval_env = eval_env_fn()
            
        #     obs, _ = eval_env.reset()
        #     for _ in range(MAX_EPISODE_STEPS):
        #         action, _ = gcn_model.predict(obs, deterministic=True)
        #         obs, reward, terminated, truncated, _ = eval_env.step(action)
        #         time.sleep(0.02) # 让动作慢下来，接近 50FPS 的视觉效果
        #         if terminated or truncated: break
        #     eval_env.close()

        print(f"⚡ 触发内环共同进化：大脑开始在线轮流适应新一代精锐...")
        env_fns = [make_evo_env(elite.lengths) for elite in elites]
        env_fns.extend([make_evo_env([0.10]), make_evo_env([0.10, 0.10]), make_evo_env([0.10, 0.10, 0.10])])
        
        # 🚀 核心修复：将总训练步数平摊给每一个环境，保持 n_envs = 1 轮流特训
        steps_per_env = RL_STEPS_PER_GEN // max(1, len(env_fns))
        
        for idx, env_fn in enumerate(env_fns):
            train_env = DummyVecEnv([env_fn])
            gcn_model.set_env(train_env)
            gcn_model.learn(total_timesteps=steps_per_env, log_interval=500)
            train_env.close()
        
        print("🧬 提取优良基因，开始繁衍下一代...")
        new_population = [RobotIndividual(elite.lengths.copy()) for elite in elites]
        while len(new_population) < POPULATION_SIZE:
            new_population.append(RobotIndividual(mutate_genome(random.choice(elites).lengths)))
            
        population = new_population
        gcn_model.save(os.path.join(evo_ckpt_dir, f"brain_gen_{gen+1}"))

    print("\n🎉 达尔文计划圆满结束！你得到了能够驾驭万物形态的终极大脑！")