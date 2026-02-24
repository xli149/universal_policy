import numpy as np
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box

class ReacherEnv(MujocoEnv, utils.EzPickle):
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, xml_file="reacher.xml", frame_skip=2, **kwargs):
        utils.EzPickle.__init__(self, xml_file, frame_skip, **kwargs)
        
        # 定义一个简单的 Box，仅用于通过 MujocoEnv 的基类校验
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
        
        # ✅ 核心改变：彻底剔除 "+10 悬停奖金"！
        # 官方 v5 根本没有成功奖金，只有“没碰到球时的扣分”。
        # 它必须为了【少扣分】而拼命飞向红球，并为了【不扣动作分】而安静停下。
        
        if self.render_mode == "human": self.render()
        
        # 永远不提前 terminated！不管碰没碰到，必须干满 50 帧。
        return observation, reward, False, False, reward_info

    def _get_dist(self):
        return np.linalg.norm(self.get_body_com("fingertip")[:2] - self.get_body_com("target")[:2])

    def _get_rew(self, action):
        dist = self._get_dist()
        
        # 1. 距离惩罚（最纯粹的物理指引）
        reward_dist = -dist 
        
        # 🚨 核心改变：剔除“进度奖励”
        # 进度奖励容易引发局部最优（来回震荡刷分），官方 v5 不需要它。
        
        # ✅ 核心改变：将动作惩罚放大 10 倍！（解决“电风扇疯狂转圈”的元凶）
        # 从 -0.01 修改为 -0.1。瞎转圈会带来极其惨重的扣分！
        reward_ctrl = -0.1 * np.square(action).sum()

        # 最终奖励就是极简的物理反馈
        reward = reward_dist + reward_ctrl
        
        return reward, {"dist": dist, "reward_dist": reward_dist, "reward_ctrl": reward_ctrl}

    def reset_model(self):
        qpos = self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq) + self.init_qpos
        while True:
            self.goal = self.np_random.uniform(low=-0.2, high=0.2, size=2)
            if np.linalg.norm(self.goal) < 0.2: break
        qpos[-2:] = self.goal
        qvel = self.init_qvel + self.np_random.uniform(low=-0.005, high=0.005, size=self.model.nv)
        qvel[-2:] = 0
        self.set_state(qpos, qvel)
        
        # 🚨 剔除了 self.prev_dist，因为不再需要计算进度了。
        return self._get_obs()

    def _get_obs(self):
        # 原汁原味保留，这些数据足够 Wrapper 提取相对位置了
        return {
            "qpos": self.data.qpos.flat[:self.n_joints].copy(),
            "qvel": self.data.qvel.flat[:self.n_joints].copy(),
            "target": self.get_body_com("target")[:2].copy(),
            "fingertip": self.get_body_com("fingertip")[:2].copy(),
        }