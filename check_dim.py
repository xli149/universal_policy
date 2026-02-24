import mujoco
import numpy as np

# 替换为你想要检查的 XML 路径
XML_PATH = "/Users/chrislee/Documents/mujoco_test/gymnasium_env/envs/reacher_2j.xml" 

def check_mujoco_dims(xml_path):
    print(f"\n{'='*20} 检查模型: {xml_path} {'='*20}")
    
    # 加载模型
    model = mujoco.MjModel.from_xml_path(xml_path)
    
    print(f"1. 广义坐标总长度 (qpos size): {model.nq}")
    print(f"2. 广义速度总长度 (qvel size): {model.nv}")
    print(f"3. 关节详细信息:")
    
    # 遍历所有关节
    for i in range(model.njnt):
        # 获取关节名称
        jnt_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        # 获取该关节在 qpos 中的起始地址
        qpos_addr = model.jnt_qposadr[i]
        # 获取该关节的类型 (0: Free, 1: Ball, 2: Slide, 3: Hinge)
        jnt_type = model.jnt_type[i]
        
        type_map = {0: "Free", 1: "Ball", 2: "Slide", 3: "Hinge"}
        
        print(f"   - 关节 [{i}] 名字: {jnt_name:10} | 类型: {type_map.get(jnt_type):6} | qpos 起始索引: {qpos_addr}")

    # 模拟一次 _get_obs 里的逻辑
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    
    print(f"\n4. 模拟 Python 切片 [2:] 的结果:")
    qpos_all = data.qpos.flatten()
    sliced = qpos_all[2:]
    print(f"   qpos 全长: {len(qpos_all)} | 切片 [2:] 长度: {len(sliced)}")
    print(f"   切片内容包含: {sliced}")

if __name__ == "__main__":
    check_mujoco_dims(XML_PATH)