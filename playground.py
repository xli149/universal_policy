import gymnasium
import gymnasium_env
from gymnasium.wrappers import TimeLimit

max_episode_steps = 100
env = gymnasium.make("gymnasium_env/Reacher2D-v0", xml_file ="/Users/chrislee/Documents/mujoco_test/gymnasium_env/envs/reacher_5j.xml", render_mode="human", )

# env = gymnasium.make("gymnasium_env/Reacher2D-v0",xml_file="./gymnasium_env/envs/test2.xml", render_mode="human", )
env = TimeLimit(env, max_episode_steps=max_episode_steps)
obs, info = env.reset()
print(f"obs is: {obs}")

for _ in range(10000):
    action = env.action_space.sample()  # Replace with your agent's action
    obs, reward, terminated, truncated, info = env.step(action)
    # print(f"reward: {reward}, terminated: {terminated}，truncated: {truncated} info: {info['dist_to_target']}")

    if terminated:
        obs, info = env.reset()

env.close()