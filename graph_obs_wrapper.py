import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box, Dict

class PaddedGraphObsWrapper(gym.ObservationWrapper):
    def __init__(self, env, max_joints=10, n_arm_joints=2):
        super().__init__(env)
        self.max_joints = max_joints
        self.n_arm_joints = n_arm_joints 
        
        self.body_ids = [env.unwrapped.model.body(f"body{i}").id for i in range(self.n_arm_joints)]
        
        # ==========================================
        # 🚀 极其巧妙的物理提取法：获取每节连杆的真实长度！
        # 在我们的 XML 中，下一节 body 的相对 X 坐标，正是当前连杆的长度 L。
        # ==========================================
        self.link_lengths = np.zeros(self.max_joints, dtype=np.float32)
        for i in range(self.n_arm_joints):
            # 找到当前连杆的“子节点”名称
            if i == self.n_arm_joints - 1:
                next_body_name = "fingertip"
            else:
                next_body_name = f"body{i+1}"
            
            # 获取该子节点的 ID
            next_body_id = env.unwrapped.model.body(next_body_name).id
            
            # 读取它的相对坐标 (x, y, z)，其中 x 即为当前连杆长度 L
            length_L = env.unwrapped.model.body_pos[next_body_id][0]
            self.link_lengths[i] = length_L
            
        print(f"✅ 物理引擎提取连杆长度成功: {self.link_lengths[:self.n_arm_joints]}")
        
        # 生成静态 mask
        self.static_mask = np.zeros(self.max_joints, dtype=np.int8)
        self.static_mask[:self.n_arm_joints] = 1
        
        self.observation_space = Dict({
            # 🚀 修改 1：shape 从 5 变成 6
            "nodes": Box(low=-np.inf, high=np.inf, shape=(self.max_joints, 6), dtype=np.float32),
            "mask": Box(low=0, high=1, shape=(self.max_joints,), dtype=np.int8),
            "global": Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32),
        })

    def observation(self, obs):
        data = self.env.unwrapped.data
        
        # 🚀 修改 2：初始化 6 个维度的空矩阵
        nodes = np.zeros((self.max_joints, 6), dtype=np.float32)
        
        qpos = data.qpos.flatten()[:self.n_arm_joints]
        qvel = data.qvel.flatten()[:self.n_arm_joints]
        
        target_xy = self.env.unwrapped.get_body_com("target")[:2]
        fingertip_xy = self.env.unwrapped.get_body_com("fingertip")[:2]
        
        nodes[:self.n_arm_joints, 0] = np.cos(qpos)
        nodes[:self.n_arm_joints, 1] = np.sin(qpos)
        nodes[:self.n_arm_joints, 2] = qvel * 0.1
        
        all_joint_xy = data.xpos[self.body_ids, :2]
        nodes[:self.n_arm_joints, 3:5] = all_joint_xy - target_xy

        # 🚀 修改 3：将之前提取缓存好的连杆长度，直接拼接到第 6 个维度
        nodes[:self.n_arm_joints, 5] = self.link_lengths[:self.n_arm_joints]

        vec = fingertip_xy - target_xy
        global_feat = np.array([target_xy[0], target_xy[1], vec[0], vec[1]], dtype=np.float32)
        
        return {"nodes": nodes, "mask": self.static_mask, "global": global_feat}


class PaddedActionWrapper(gym.ActionWrapper):
    def __init__(self, env, max_joints=10, n_arm_joints=2):
        super().__init__(env)
        self.max_joints = max_joints
        self.n_arm_joints = n_arm_joints
        
        low = np.full(self.max_joints, -1.0, dtype=np.float32)
        high = np.full(self.max_joints, 1.0, dtype=np.float32)
        self.action_space = Box(low=low, high=high, dtype=np.float32)

    def action(self, act):
        return act[:self.n_arm_joints]