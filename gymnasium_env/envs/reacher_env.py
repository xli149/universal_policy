import numpy as np
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box

class ReacherEnv(MujocoEnv, utils.EzPickle):
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, xml_file="reacher.xml", frame_skip=2, **kwargs):
        utils.EzPickle.__init__(self, xml_file, frame_skip, **kwargs)
        
        # 定义一个简单的 Box，仅用于通过 MujocoEnv 的基类校验
        # 实际的 Dict 空间由 Wrapper 定义
        dummy_space = Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float64)
        
        MujocoEnv.__init__(self, xml_file, frame_skip, observation_space=dummy_space, **kwargs)

        self.max_joints = 10
        # ✅ 让环境承认 10 维动作空间，匹配网络输出
        self.action_space = Box(low=-1.0, high=1.0, shape=(self.max_joints,), dtype=np.float32)
        
        self.n_joints = self.model.nu 
        self.success_threshold = 0.10 

    def step(self, action):
        actual_action = action[:self.n_joints]
        self.do_simulation(actual_action, self.frame_skip)
        
        observation = self._get_obs()
        reward, reward_info = self._get_rew(actual_action)
        
        dist = self._get_dist()
        
        # ✅ 核心改变 1：永远不提前 terminated！不管碰没碰到，必须干满 50 帧。
        terminated = False 
        
        # ✅ 核心改变 2：“打卡工资”变成“驻留时薪”
        # 只要手尖在红球里，【每一帧】都给 +10 分！
        # 如果它第一秒就到了并黏住，一局能拿几百分的暴利！
        if dist < self.success_threshold:
            reward += 10.0 
            
        if self.render_mode == "human": self.render()
        return observation, reward, terminated, False, reward_info

    def _get_dist(self):
        return np.linalg.norm(self.get_body_com("fingertip")[:2] - self.get_body_com("target")[:2])

    def _get_rew(self, action):
        dist = self._get_dist()
        
        # 距离越远，依然会有小额扣分，用来指引方向
        reward_dist = -dist 
        
        # 进度奖励保留，让它在没碰到球之前能顺着气味找过去
        reward_progress = 0.0
        if self.prev_dist is not None:
            reward_progress = (self.prev_dist - dist) * 10.0 
            
        reward_ctrl = -0.01 * np.square(action).sum()

        # 🚨 删除了 step_penalty。不需要皮鞭了，前方的“每帧 +10 分”就是最强磁铁。

        self.prev_dist = dist
        return reward_dist + reward_progress + reward_ctrl, {"dist": dist}

    def reset_model(self):
        qpos = self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq) + self.init_qpos
        while True:
            self.goal = self.np_random.uniform(low=-0.2, high=0.2, size=2)
            if np.linalg.norm(self.goal) < 0.2: break
        qpos[-2:] = self.goal
        qvel = self.init_qvel + self.np_random.uniform(low=-0.005, high=0.005, size=self.model.nv)
        qvel[-2:] = 0
        self.set_state(qpos, qvel)
        
        # ✅ 致命 Bug 修复：每局开始前，必须把 prev_dist 设为当前的绝对初始距离！
        # 否则进度奖励会发生极其离谱的“跨局污染”。
        self.prev_dist = self._get_dist()
        
        return self._get_obs()

    def _get_obs(self):
        return {
            "qpos": self.data.qpos.flat[:self.n_joints].copy(),
            "qvel": self.data.qvel.flat[:self.n_joints].copy(),
            "target": self.get_body_com("target")[:2].copy(),
            "fingertip": self.get_body_com("fingertip")[:2].copy(),
        }