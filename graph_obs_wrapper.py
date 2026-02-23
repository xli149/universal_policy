import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box, Dict

def obs_to_nodes(qpos, qvel, n_joints, max_joints, target_xy, fingertip_xy, env):
    nodes = np.zeros((max_joints, 5), dtype=np.float32) 
    mask = np.zeros((max_joints,), dtype=np.int8)

    for i in range(n_joints):
        mask[i] = 1
        nodes[i, 0] = np.cos(qpos[i])
        nodes[i, 1] = np.sin(qpos[i])
        nodes[i, 2] = qvel[i] * 0.1
        
        body_name = env.unwrapped.model.body(i+1).name
        joint_xy = env.unwrapped.get_body_com(body_name)[:2]
        rel_target = joint_xy - target_xy
        nodes[i, 3] = rel_target[0]
        nodes[i, 4] = rel_target[1]

    vec = fingertip_xy - target_xy
    global_feat = np.array([target_xy[0], target_xy[1], vec[0], vec[1]], dtype=np.float32)
    return nodes, mask, global_feat

class PaddedGraphObsWrapper(gym.ObservationWrapper):
    def __init__(self, env, max_joints=10):
        super().__init__(env)
        self.max_joints = max_joints
        self.n_joints = env.unwrapped.n_joints
        
        # ✅ 使用显式的 Dict 空间，防止 SB3 提取特征时报错
        self.observation_space = Dict({
            "nodes": Box(low=-np.inf, high=np.inf, shape=(max_joints, 5), dtype=np.float32),
            "mask": Box(low=0, high=1, shape=(max_joints,), dtype=np.int8),
            "global": Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32),
        })

    def observation(self, obs):
        nodes, mask, global_feat = obs_to_nodes(
            obs["qpos"], obs["qvel"], self.n_joints, self.max_joints, 
            obs["target"], obs["fingertip"], self.env
        )
        return {"nodes": nodes, "mask": mask, "global": global_feat}