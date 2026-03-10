# env_sanity_check_visual.py
from co_evolution_main import build_reacher_xml
import mujoco
import mujoco.viewer
import time
import numpy as np

test_cases = [
    [0.1],                       # 1关节，极短
    [0.1, 0.1, 0.1],             # 3关节，标准
    [0.15, 0.15, 0.15, 0.15, 0.15], # 5关节，超长
    [0.05] * 10                  # 10关节，极细长蛇
]

print("=== 🌍 创世环境合理性检查 & 可视化 ===\n")

for i, lengths in enumerate(test_cases):
    n_j = len(lengths)
    
    # 🚀 核心修改：调用最新的模块化造物主，注入物理防爆护盾！
    xml_str = build_reacher_xml(
        model_name=f"test_case_{n_j}j",
        lengths=lengths,
        link_radius=0.015,
        joint_range=(-2.2, 2.2), # 防止死锁
        gear=40.0,               # 降低马力
        arena_margin=1.35,  
        target_ratio=0.9,
    )
    
    try:
        # 1. 逻辑审计
        model = mujoco.MjModel.from_xml_string(xml_str)
        data = mujoco.MjData(model)
        arm_len = sum(lengths)
        # target_x 的范围在倒数第二个关节
        target_range = model.jnt_range[-2][1] 
        
        print(f"🎬 Case {i+1} [{n_j} 关节]:")
        print(f"   - 总臂长: {arm_len:.2f}m")
        print(f"   - 目标球活动半径: {target_range:.2f}m")
        
        # 逻辑判定
        if target_range <= arm_len:
            status = "✅ 目标可达"
        else:
            status = "❌ 警告：目标超出手臂极限"
        print(f"   - 状态: {status}")

        # 2. 物理可视化渲染
        print(f"   - [渲染中] 请观察弹出窗口 (3秒后自动切换下一个)...")
        
        with mujoco.viewer.launch_passive(model, data) as viewer:
            # 渲染起始位置
            mujoco.mj_forward(model, data)
            
            start_time = time.time()
            while viewer.is_running() and (time.time() - start_time < 3.0):
                step_start = time.time()
                
                # 让机械臂像章鱼一样动一动，方便观察关节连接是否正常
                # 给每个电机一个正弦波信号
                data.ctrl[:] = 0.5 * np.sin(time.time() * 3 + np.arange(n_j))
                
                mujoco.mj_step(model, data)
                viewer.sync()
                
                # 控制渲染帧率
                time_until_next_step = model.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)
            viewer.close()
        time.sleep(0.5)
        print(f"   - Case {i+1} 检查完毕。\n")

    except Exception as e:
        print(f"   - 🚨 Case {i+1} 发生毁灭性错误: {e}\n")

print("=== 🏁 所有环境检查结束 ===")