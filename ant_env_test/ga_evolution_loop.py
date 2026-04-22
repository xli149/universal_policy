import random
import os
import time
import pickle
# from ant_env_test.co_evolution_main import build_asymmetric_ant_xml
from ant_env_test.co_evolution_main import build_mutant_ant_xml
import gymnasium as gym
from gymnasium_env.envs.ant_env_v5 import AntEnv
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv

from ant_env_test.graph_obs_wrapper import AntGraphObsWrapper, AntActionWrapper
from ant_env_test.masked_policy import MaskedGraphSACPolicy

CHECKPOINT_PATH = "ga_checkpoint.pkl"

def save_checkpoint(generation, population, historical_best_score):
    """将训练进度保存到硬盘"""
    data = {
        "generation": generation,
        "population": population,
        "historical_best_score": historical_best_score
    }
    with open(CHECKPOINT_PATH, "wb") as f:
        pickle.dump(data, f)
    print(f"\n💾 进度已存档：第 {generation + 1} 代演化已妥善保存。")

def load_checkpoint():
    """从硬盘读取训练进度"""
    if os.path.exists(CHECKPOINT_PATH):
        try:
            with open(CHECKPOINT_PATH, "rb") as f:
                data = pickle.load(f)
            print(f"\n项目重启！检测到存档：从第 {data['generation'] + 2} 代继续开始...")
            return data
        except Exception as e:
            print(f"存档损坏或读取失败: {e}")
    return None


def evaluate_fitness(genome, base_model_path):
    """⚖️ 试炼场：带有动态微调和防遗忘机制的适应度评估"""
    num_legs = len(genome)
    xml_path = "./temp_xmls/eval_ga_ant.xml"
    os.makedirs("./temp_xmls", exist_ok=True)
    with open(xml_path, "w") as f:
        # ⚠️ 前提提醒：请确保你已经按照之前的讨论，
        # 在 build_asymmetric_ant_xml 函数里加入了“自适应出生高度(spawn_z)”的修复！
        f.write(build_mutant_ant_xml(genome))
        
    # def _init_env():
    #     env = AntEnv(xml_file=xml_path, render_mode=None, reset_noise_scale=0.0)
    #     env = gym.wrappers.TimeLimit(env, max_episode_steps=1000)
    #     env = AntGraphObsWrapper(env, num_legs=num_legs, max_legs=8)
    #     return AntActionWrapper(env, num_legs=num_legs, max_legs=8)

    def _init_env():
        env = AntEnv(
            xml_file=xml_path, 
            render_mode=None, 
            reset_noise_scale=0.0,
            # 🚨 必须与有效环境保持绝对一致！
            healthy_z_range=(0.13, 1.5),  
            healthy_reward=1.0,           
            forward_reward_weight=3.0,    
            ctrl_cost_weight=0.05         # 降低电费，防止它因为抽搐被扣到 -300 分
        )
        env = gym.wrappers.TimeLimit(env, max_episode_steps=1000)
        env = AntGraphObsWrapper(env, num_legs=num_legs, max_legs=8)
        return AntActionWrapper(env, num_legs=num_legs, max_legs=8)
   
    test_env = _init_env()

    # ----------------------------------------------------
    # 💉 核心修复 1：注入“返老还童针”
    # 强制重置好奇心 (ent_coef) 促使探索，降低学习率 (learning_rate) 防止遗忘
    # ----------------------------------------------------
    custom_objects = {
        'policy_class': MaskedGraphSACPolicy,
        'ent_coef': 'auto_0.1',  
        'learning_rate': 1e-4    
    }

    model = SAC.load(
        base_model_path,
        env=test_env,
        custom_objects=custom_objects
    )

    # ================= 🎯 阶段 1：零样本摸底考试 =================
    # deterministic=True 保证这里绝对没有随机探索，只测纯粹的肌肉记忆！
    zero_shot_score = 0
    step_count = 0
    obs, info = test_env.reset()
    for step in range(200):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = test_env.step(action)
        zero_shot_score += reward
        step_count += 1

        if terminated or truncated:
            break

    print(f"   [摸底测试] 存活 {step_count} 步，初始得分: {zero_shot_score:.2f}")

    # ================= 🔧 阶段 2：动态微调 =================
    if zero_shot_score < 10:
        print("   🔧 躯体不兼容，开启 2000 步高探索康复训练...")
        # learn() 底层会自动切入随机探索模式，上面注入的 ent_coef 发威！
        model.learn(total_timesteps=2000)
    else:
        print("   ✨ 天赋异禀！免试微调！")

    # ================= 🏆 阶段 3：终局考核 =================
    final_score = 0
    step_count = 0              
    obs, info = test_env.reset() # 🚨 核心修复：重置环境与计数器！

    for _ in range(1000):
        # 再次切回 deterministic=True，冷酷无情地验收成果
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = test_env.step(action)
        final_score += reward
        step_count += 1
        
        if terminated or truncated:
            reason = "💀 摔倒翻车" if terminated else "⏱️ 坚持到了最后"
            print(f"   [终局测试] 存活 {step_count} 步 | 总得分: {final_score:.2f} | 原因: {reason}")
            break

    test_env.close()
    return final_score

def visualize_elites(elites, base_model_path, generation):
    """📺 汇报表演：可视化展示本代最强的 4 只生物"""
    print(f"\n🎬 开启第 {generation} 代精英汇报表演...")
    
    for i, genome in enumerate(elites):
        num_legs = len(genome)
        print(f"   🎥 正在展示 [精英 {i+1}/4] (腿数: {num_legs})...")
        
        # 1. 临时生成 XML
        xml_path = f"./temp_xmls/viz_gen{generation}_elite{i}.xml"
        with open(xml_path, "w") as f:
            f.write(build_mutant_ant_xml(genome))
            
        # 2. 初始化带渲染的环境
        def _init_viz():
            # 💡 注意：这里 render_mode 设为 "human" 才能看到窗口
            env = AntEnv(xml_file=xml_path, render_mode="human", reset_noise_scale=0.0)
            env = gym.wrappers.TimeLimit(env, max_episode_steps=500) # 表演 500 步够了
            env = AntGraphObsWrapper(env, num_legs=num_legs, max_legs=8)
            return AntActionWrapper(env, num_legs=num_legs, max_legs=8)
        
        viz_env = _init_viz()
        
        # 3. 加载模型 (可视化不需要微调，直接看它微调后的表现)
        # 注意：这里我们无法直接拿到微调后的权重，
        # 如果你想看微调后的效果，需要在 evaluate_fitness 里把 model return 出来或者 save 下来。
        # 暂时我们直接用基座大脑看它的零样本表现，或者在下面接入微调逻辑。
        model = SAC.load(base_model_path, env=viz_env, custom_objects={'policy_class': MaskedGraphSACPolicy})
        
        # 4. 跑一个回合的表演
        obs, info = viz_env.reset()
        for _ in range(500):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = viz_env.step(action)
            time.sleep(0.01) # 💡 加一点延迟，防止跑太快看不清动作
            if terminated or truncated:
                break
        
        viz_env.close()
    print("   ✅ 表演结束，准备进入下一代演化...")


# def generate_random_genome(min_leg=4, max_leg=8):
#     """🧬 造物主：生成随机变异体基因"""
#     num_legs = random.randint(min_leg, max_leg)
#     genome = []

#     for i in range(num_legs):
#         base_angle = (360/num_legs) * i
#         noise = random.uniform(-15, 15)
#         angle = base_angle + noise

#         thigh_len = random.uniform(0.1, 0.5)
#         calf_len = random.uniform(0.1, 0.5)

#         hip_min = random.uniform(-40, -10)
#         hip_max = random.uniform(10, 40)
#         ankle_min = random.uniform(10, 40)
#         ankle_max = random.uniform(60, 100)

#         leg_gene = [angle, thigh_len, calf_len, hip_min, hip_max, ankle_min, ankle_max]
#         genome.append(leg_gene)

#     return genome

# import random

def generate_random_genome(min_leg=4, max_leg=8):
    """
    🧬 造物主 4.0：残酷淘汰实验版 (强制 4, 6, 8 偶数足 + 辐射对称)
    """
    # 只允许 4, 6, 8 条腿出生
    num_legs = random.randint(min_leg, max_leg)

    genome = []

    for i in range(num_legs):
        # 辐射对称基准角度
        base_angle = (360 / num_legs) * i
        noise = random.uniform(-5, 5) 
        angle = base_angle + noise

        # 初始腿长设定在一个相对保守、相近的区间
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

# def generate_random_genome(min_leg=4, max_leg=8):
#     # 临时写死，剥夺随机性，用于测试！
#     num_legs = 6 
#     genome = []
#     for i in range(num_legs):
#         angle = (360 / num_legs) * i
#         # 全都锁死在 0.2 和标准角度
#         leg_gene = [angle, 0.2, 0.2, -30, 30, 30, 70] 
#         genome.append(leg_gene)
#     return genome


def mutate(genome, mutation_rate=0.2):
    """✂️ 基因手术刀：变异"""
    mutated_genome = []
    for leg in genome:
        new_leg = list(leg)

        if random.random() < mutation_rate:
            new_leg[1] = leg[1] * random.uniform(0.8, 1.5) # 大腿长

        if random.random() < mutation_rate:
            new_leg[2] = leg[2] * random.uniform(0.8, 1.5) # 小腿长

        if random.random() < mutation_rate:
            new_leg[3] = leg[3] * random.uniform(0.8, 1.5) # 髋关节 min
            new_leg[4] = leg[4] * random.uniform(0.8, 1.5) # 髋关节 max
            
        # ----------------------------------------------------
        # 💉 核心修复 2：补全缺失的脚踝变异逻辑！
        # ----------------------------------------------------
        if random.random() < mutation_rate:
            new_leg[5] = leg[5] * random.uniform(0.8, 1.5) # 踝关节 min
            new_leg[6] = leg[6] * random.uniform(0.8, 1.5) # 踝关节 max
        
        mutated_genome.append(new_leg)

    return mutated_genome


def crossover(parent1, parent2):
    """💞 基因交配：父母基因重组"""
    target_legs = random.choice([len(parent1), len(parent2)])
    pool = parent1 + parent2
    random.shuffle(pool)
    
    child_genome = []
    for i in range(target_legs):
        leg = list(pool[i])
        base_angle = (360/target_legs) * i
        noise = random.uniform(-15, 15)
        angle = base_angle + noise
        leg[0] = angle
        child_genome.append(leg)
    
    return child_genome


if __name__ == "__main__":
    # =====================================================================
    # 🌍 核心修复 3：真正的达尔文演化主循环 (The Darwinian Engine)
    # =====================================================================
    
    # 🚨 请确保这里的路径指向你本地真实有效的满级模型！
    MODEL_PATH = "./checkpoints/generalist_base_brain_normal/gcn_generalist_base_1400000_steps" 
    
    POPULATION_SIZE = 10  # 种群规模：每一代 10 只怪物
    GENERATIONS = 20      # 进化纪元：模拟 20 代更迭


    checkpoint = load_checkpoint()

    if checkpoint:
        # 如果有存档，恢复数据
        start_generation = checkpoint["generation"] + 1
        population = checkpoint["population"]
        historical_best_score = checkpoint["historical_best_score"]
    else:
        # 如果没存档，初始化
        start_generation = 0
        print("🌍 [创世大爆炸] 正在初始化第一代随机种群...")
        population = [generate_random_genome() for _ in range(POPULATION_SIZE)]
        historical_best_score = -float('inf')

    
    for gen in range(start_generation, GENERATIONS):
        print(f"\n{'='*50}")
        print(f"🧬 第 {gen + 1} 代进化纪元开启")
        print(f"{'='*50}")
        
        # 1. 考官打分阶段
        scored_population = []
        for i, genome in enumerate(population):
            print(f"\n👉 正在评估 [变异体 {i+1}/{POPULATION_SIZE}] (腿数: {len(genome)})")
            
            # 把怪物送进试炼场
            score = evaluate_fitness(genome, MODEL_PATH)
            scored_population.append((score, genome))
            
        # 2. 优胜劣汰阶段 (按分数从高到低排序)
        scored_population.sort(key=lambda x: x[0], reverse=True)
        
        # 提取本代排名前 4 的精英
        elites = [item[1] for item in scored_population[:4]]

        visualize_elites(elites, MODEL_PATH, gen + 1)

        best_score = scored_population[0][0]
        best_genome = elites[0]
        
        if best_score > historical_best_score:
            historical_best_score = best_score
            print(f"\n🌟 [历史突破] 诞生了全新的完美物种！历史最高分刷新为: {historical_best_score:.2f} (腿数: {len(best_genome)})")
        else:
            print(f"\n🏆 [本代结算] 本代最强王者得分: {best_score:.2f} (腿数: {len(best_genome)})")
        
        # 3. 繁衍阶段：用精英填满下一代
        new_population = list(elites) # 前 4 名精英直接免死进入下一代
        
        print(f"💞 精英正在交配繁衍 {POPULATION_SIZE - len(elites)} 个新后代...")
        while len(new_population) < POPULATION_SIZE:
            parent1 = random.choice(elites)
            parent2 = random.choice(elites)
            
            child = crossover(parent1, parent2)
            child = mutate(child, mutation_rate=0.2)
            
            new_population.append(child)
            
        # 4. 世代更迭
        population = new_population

        save_checkpoint(gen, population, historical_best_score)

    print("\n🎉 整个演化模拟彻底完成！达尔文引擎已停机。")
    print(f"👑 历史最强物种得分: {historical_best_score:.2f}")