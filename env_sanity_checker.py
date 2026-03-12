# env_sanity_check_visual.py
from co_evolution_main import build_reacher_xml
import mujoco
import mujoco.viewer
import time
import numpy as np

# 测试用例：从极短到极长
test_cases = [
    [0.1],                       # 1关节，极短
    [0.1, 0.1, 0.1],             # 3关节，标准
    [0.15, 0.15, 0.15, 0.15, 0.15], # 5关节，超长
    [0.05] * 10                  # 10关节，极细长蛇
]

print("=== 🌍 创世环境（避障版）合理性检查 & 可视化 ===\n")

for i, lengths in enumerate(test_cases):
    n_j = len(lengths)
    
    # 调用最新的造物主，里面现在包含了那根绿色的柱子！
    xml_str = build_reacher_xml(
        model_name=f"test_case_{n_j}j",
        lengths=lengths,
        link_radius=0.015,
        joint_range=(-2.2, 2.2),
        gear=40.0,               
        # arena_margin=1.35,  
        # target_ratio=0.9,
    )
    
    try:
        # 1. 逻辑审计
        model = mujoco.MjModel.from_xml_string(xml_str)
        data = mujoco.MjData(model)
        arm_len = sum(lengths)
        target_range = model.jnt_range[-2][1] 
        
        print(f"🎬 Case {i+1} [{n_j} 关节]:")
        print(f"   - 总臂长: {arm_len:.2f}m")
        print(f"   - 目标球活动半径: {target_range:.2f}m")
        print(f"   - 状态: {'✅ 目标可达' if target_range <= arm_len else '❌ 警告：目标超出手臂极限'}")

        # 2. 物理可视化渲染
        print(f"   - [渲染中] 请寻找绿色的【叹息之柱】并观察碰撞 (5秒后自动切换)...")
        
        with mujoco.viewer.launch_passive(model, data) as viewer:
            mujoco.mj_forward(model, data)
            
            start_time = time.time()
            # 延长时间到 5 秒，让你看清碰撞
            while viewer.is_running() and (time.time() - start_time < 5.0):
                step_start = time.time()
                
                # 🚀 加大挥舞力度！强迫它向右上方（柱子所在方向）扫荡
                # 用 cos 和 sin 的组合让它在场子里疯狂甩动
                wave = np.sin(time.time() * 5 + np.arange(n_j))
                data.ctrl[:] = 1.0 * wave 
                
                mujoco.mj_step(model, data)
                viewer.sync()
                
                time_until_next_step = model.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)
            viewer.close()
        time.sleep(0.5)
        print(f"   - Case {i+1} 检查完毕。\n")

    except Exception as e:
        print(f"   - 🚨 Case {i+1} 发生毁灭性错误: {e}\n")

print("=== 🏁 所有环境检查结束 ===")