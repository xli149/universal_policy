import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box, Dict

class PaddedGraphObsWrapper(gym.ObservationWrapper):
    # 🚀 增加 n_arm_joints 参数，明确告诉它真实手臂有几个关节
    def __init__(self, env, max_joints=10, n_arm_joints=2):
        super().__init__(env)
        self.max_joints = max_joints
        self.n_arm_joints = n_arm_joints 
        
        # 🚀 修复：只抓取前 n_arm_joints 个身体连杆的 ID
        self.body_ids = [env.unwrapped.model.body(i+1).id for i in range(self.n_arm_joints)]
        
        # 生成静态 mask (只有前两个节点是 1，后面 8 个全被屏蔽)
        self.static_mask = np.zeros(self.max_joints, dtype=np.int8)
        self.static_mask[:self.n_arm_joints] = 1
        
        self.observation_space = Dict({
            "nodes": Box(low=-np.inf, high=np.inf, shape=(self.max_joints, 5), dtype=np.float32),
            "mask": Box(low=0, high=1, shape=(self.max_joints,), dtype=np.int8),
            "global": Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32),
        })

    def observation(self, obs):
        data = self.env.unwrapped.data
        nodes = np.zeros((self.max_joints, 5), dtype=np.float32)
        
        # 🚀 修复：精准切片！只取前 2 个真正关节的角度和速度
        qpos = data.qpos.flatten()[:self.n_arm_joints]
        qvel = data.qvel.flatten()[:self.n_arm_joints]
        
        target_xy = self.env.unwrapped.get_body_com("target")[:2]
        fingertip_xy = self.env.unwrapped.get_body_com("fingertip")[:2]
        
        # 纯向量化赋值：现在这里面绝对只有真实的关节角度了
        nodes[:self.n_arm_joints, 0] = np.cos(qpos)
        nodes[:self.n_arm_joints, 1] = np.sin(qpos)
        nodes[:self.n_arm_joints, 2] = qvel * 0.1
        
        # 这里的 all_joint_xy 现在只包含 body0 和 body1 的真实坐标
        all_joint_xy = data.xpos[self.body_ids, :2]
        nodes[:self.n_arm_joints, 3:5] = all_joint_xy - target_xy

        # 全局特征（包含目标位置和末端距离）
        vec = fingertip_xy - target_xy
        global_feat = np.array([target_xy[0], target_xy[1], vec[0], vec[1]], dtype=np.float32)
        
        return {"nodes": nodes, "mask": self.static_mask, "global": global_feat}


class PaddedActionWrapper(gym.ActionWrapper):
    # 🚀 同样增加 n_arm_joints 参数
    def __init__(self, env, max_joints=10, n_arm_joints=2):
        super().__init__(env)
        self.max_joints = max_joints
        self.n_arm_joints = n_arm_joints
        
        # 让 SB3 以为动作空间是 10 维的
        low = np.full(self.max_joints, -1.0, dtype=np.float32)
        high = np.full(self.max_joints, 1.0, dtype=np.float32)
        self.action_space = Box(low=low, high=high, dtype=np.float32)

    def action(self, act):
        # 🚀 真正发给物理引擎时，截断后面的废弃动作，只发前 2 个！
        return act[:self.n_arm_joints]