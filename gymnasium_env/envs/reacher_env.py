import numpy as np
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box

class ReacherEnv(MujocoEnv, utils.EzPickle):
    metadata = {"render_modes": ["human", "rgb_array", "depth_array"]}

    def __init__(self, xml_file="reacher.xml", frame_skip=4, **kwargs):
        utils.EzPickle.__init__(self, xml_file, frame_skip, **kwargs)
        
        dummy_space = Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float64)
        MujocoEnv.__init__(self, xml_file, frame_skip, observation_space=dummy_space, **kwargs)

        self.max_joints = 10
        self.action_space = Box(low=-1.0, high=1.0, shape=(self.max_joints,), dtype=np.float32)
        
        self.n_joints = self.model.nu 
        self.success_threshold = 0.10 

    def step(self, action):
        actual_action = action[:self.n_joints]
        self.do_simulation(actual_action, self.frame_skip)
        
        observation = self._get_obs()
        reward, reward_info = self._get_rew(actual_action)
        
        # 🚨 物理防爆保险丝：监测极端的角速度（鞭端效应）
        qvel = self.data.qvel.flatten()[:self.n_joints]
        if np.any(np.abs(qvel) > 30.0): 
            # 强制结束当前回合，并给予极端扣分，防止物理引擎崩溃
            return observation, -100.0, True, False, {"dist": self._get_dist()}
        
        if self.render_mode == "human": self.render()
        
        # 正常情况下，必须走满回合，培养平稳控制
        return observation, reward, False, False, reward_info

    def _get_dist(self):
        return np.linalg.norm(self.get_body_com("fingertip")[:2] - self.get_body_com("target")[:2])

    def _get_rew(self, action):
        dist = self._get_dist()
        reward_dist = -dist 
        
        # 放大惩罚，终结转圈
        reward_ctrl = -0.1 * np.square(action).sum()

        reward = reward_dist + reward_ctrl
        return reward, {"dist": dist, "reward_dist": reward_dist, "reward_ctrl": reward_ctrl}

    def reset_model(self):
        qpos = self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq) + self.init_qpos
        
        # 🚀 核心修复：从当前的 XML 里动态读取物理墙的大小！
        # 在我们的设定里，倒数第二个和倒数第一个关节必定是 target_x 和 target_y
        target_x_range = self.model.jnt_range[-2] 
        target_y_range = self.model.jnt_range[-1] 
        
        # 找出最大合法生成半径
        max_radius = min(target_x_range[1], target_y_range[1]) 
        
        while True:
            # 乖乖在 XML 允许的围栏内部生成目标坐标，绝不越界！
            goal_x = self.np_random.uniform(low=target_x_range[0], high=target_x_range[1])
            goal_y = self.np_random.uniform(low=target_y_range[0], high=target_y_range[1])
            self.goal = np.array([goal_x, goal_y])
            
            # 保证目标在这个合法的圆内
            if np.linalg.norm(self.goal) < max_radius: 
                break
                
        qpos[-2:] = self.goal
        qvel = self.init_qvel + self.np_random.uniform(low=-0.005, high=0.005, size=self.model.nv)
        qvel[-2:] = 0
        self.set_state(qpos, qvel)
        
        return self._get_obs()

    def _get_obs(self):
        return {
            "qpos": self.data.qpos.flat[:self.n_joints].copy(),
            "qvel": self.data.qvel.flat[:self.n_joints].copy(),
            "target": self.get_body_com("target")[:2].copy(),
            "fingertip": self.get_body_com("fingertip")[:2].copy(),
        }