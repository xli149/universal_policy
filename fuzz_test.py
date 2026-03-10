import random
import mujoco
from co_evolution_main import build_reacher_xml

TEST_ROUNDS = 1000
success_count = 0

print(f"🌪️ 开始高强度 Fuzzing 模糊测试，目标生成 {TEST_ROUNDS} 个随机变异体...\n")

for i in range(TEST_ROUNDS):
    # 1. 动态进度条渲染 (极客版)
    percent = (i + 1) / TEST_ROUNDS
    bar_length = 40
    filled_len = int(bar_length * percent)
    bar = '█' * filled_len + '-' * (bar_length - filled_len)
    
    # 使用 \r 回到行首并原地覆盖，flush=True 强制立刻输出到屏幕
    print(f"\r🚀 测试进度: [{bar}] {i+1}/{TEST_ROUNDS} ({percent*100:.1f}%)", end="", flush=True)
    
    # 2. 像 GA 一样随机疯狂生成基因
    n_j = random.randint(1, 10)
    lengths = [random.choice([0.05, 0.10, 0.15]) for _ in range(n_j)]
    
    # 3. 扔进造物工厂
    xml_str = build_reacher_xml(
        model_name=f"fuzz_{i}",
        lengths=lengths,
        link_radius=0.015,
        joint_range=(-2.2, 2.2),
        gear=40.0,
        arena_margin=1.35,
        target_ratio=0.9,
    )
    
    try:
        # 4. 核心大考：不渲染，但让物理引擎强行解析并演算一步！
        model = mujoco.MjModel.from_xml_string(xml_str)
        data = mujoco.MjData(model)
        mujoco.mj_step(model, data)  # 演算一步，检查是否会因为穿模导致算力爆炸
        success_count += 1
    except Exception as e:
        # 如果报错，先换行，免得覆盖了进度条
        print(f"\n\n🚨 抓到致盲 Bug！")
        print(f"💥 死亡基因: {lengths}")
        print(f"💥 报错信息: {e}")
        break

print(f"\n\n✅ 测试结束！存活率: {success_count} / {TEST_ROUNDS}")
if success_count == TEST_ROUNDS:
    print("🏆 恭喜！你的环境生成器如同磐石般坚不可摧，可以放心开启进化了！")