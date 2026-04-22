import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box, Dict

class AntGraphObsWrapper(gym.ObservationWrapper):
    def __init__(self, env, num_legs, max_legs=8):
        super().__init__(env)
        self.num_legs = num_legs
        self.max_legs = max_legs
        
        # 躯干(1) + 每条腿(2个关节) = 动态节点数
        self.max_nodes = 1 + self.max_legs * 2 
        self.valid_nodes = 1 + self.num_legs * 2
        
        # 静态 Mask，屏蔽掉不存在的腿
        self.static_mask = np.zeros(self.max_nodes, dtype=np.int8)
        self.static_mask[:self.valid_nodes] = 1
        
        # 提取身体 ID 用于查询接触力 (触地检测)
        self.geom_ids = []
        for i in range(self.num_legs):
            self.geom_ids.append(self.env.unwrapped.model.body(f"leg_{i}_aux").id)
            self.geom_ids.append(self.env.unwrapped.model.body(f"leg_{i}_ankle").id)

        self.observation_space = Dict({
            "nodes": Box(low=-np.inf, high=np.inf, shape=(self.max_nodes, 4), dtype=np.float32),
            "mask": Box(low=0, high=1, shape=(self.max_nodes,), dtype=np.int8),
            "global": Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32),
        })

    def observation(self, obs):
        nodes = np.zeros((self.max_nodes, 4), dtype=np.float32)
        
        # 1. 提取全局特征 (躯干的前庭觉：Z高度, 姿态四元数前3位, XY线速度)
        torso_z = obs["torso_pos"][2]
        quat = obs["torso_quat"][1:4] 
        vx, vy = obs["qvel"][0:2]
        global_feat = np.array([torso_z, quat[0], quat[1], quat[2], vx, vy], dtype=np.float32)
        
        # 2. 提取节点特征 (跳过 root 的 7 个自由度，从 hinge 关节开始)
        qpos_hinge = obs["qpos"][7:]
        qvel_hinge = obs["qvel"][6:]
        
        # 节点 0 是 Torso (作为信息中枢，局部物理特征置0，因为它吃 global 特征)
        nodes[0] = [0.0, 0.0, 0.0, 0.0] 
        
        # 填充肢体节点
        for i in range(self.num_legs * 2):
            node_idx = i + 1
            body_id = self.geom_ids[i]
            
            # 计算接触力大小，判断是否触地
            contact_force = np.linalg.norm(obs["cfrc_ext"][body_id][:3])
            is_contact = 1.0 if contact_force > 1.0 else 0.0
            
            nodes[node_idx, 0] = qpos_hinge[i]
            nodes[node_idx, 1] = qvel_hinge[i]
            nodes[node_idx, 2] = is_contact
            nodes[node_idx, 3] = 0.1 # 暂时写死的连杆长度，后续可从 XML 提取
            
        return {"nodes": nodes, "mask": self.static_mask, "global": global_feat}


class AntActionWrapper(gym.ActionWrapper):
    def __init__(self, env, num_legs, max_legs=8):
        super().__init__(env)
        self.num_motors = num_legs * 2
        self.max_motors = max_legs * 2
        # Action space 锁定为最大马达数，方便 SB3 统一处理
        self.action_space = Box(low=-1.0, high=1.0, shape=(self.max_motors,), dtype=np.float32)

    def action(self, act):
        # 截断无用的马达动作，只把真实存在的马达指令发给物理引擎
        return act[:self.num_motors]