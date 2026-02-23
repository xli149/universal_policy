import gymnasium
import gymnasium_env
from gymnasium.wrappers import TimeLimit
from ppo import PPO
import torch
import os
from datetime import datetime
import wandb
import numpy as np
from tb_logger import TBLogger
import time


def _clear_buffer(agent):
    """尽量清空 PPO buffer，避免 eval 时 select_action 往 buffer 里塞东西导致内存增长/干扰。"""
    if not hasattr(agent, "buffer"):
        return
    for name in ["states", "actions", "logprobs", "rewards", "is_terminals"]:
        if hasattr(agent.buffer, name):
            getattr(agent.buffer, name).clear()


def evaluate_checkpoint(
    env_name: str,
    checkpoint_path: str,
    max_episode_steps: int,
    n_eval_episodes: int,
    seed: int,
    state_dim: int,
    action_dim: int,
    # PPO init args (保持和训练一致)
    lr_actor: float,
    lr_critic: float,
    gamma: float,
    K_epochs: int,
    eps_clip: float,
    has_continuous_action_space: bool,
    action_std_init: float,
    render: bool = False,
):
    """load checkpoint 后跑 n_eval_episodes，返回 mean/std/min/max。"""
    # env_name = "reacher2d_2joints"
    ENV_ID = "gymnasium_env/Reacher2D-v0"
    XML_FILE = "./gymnasium_env/envs/test2.xml"
    # 单独建一个 eval env，避免影响训练 env 的状态/随机性
    render_mode = "human" if render else None
    # eval_env = gymnasium.make(env_name, render_mode=render_mode)
    eval_env = gymnasium.make(ENV_ID, xml_file=XML_FILE, render_mode=render_mode)

    eval_env = TimeLimit(eval_env, max_episode_steps=max_episode_steps)

    # 重新建 agent 再 load，确保测到的是“保存出来的东西”
    agent = PPO(
        state_dim, action_dim,
        lr_actor, lr_critic,
        gamma, K_epochs, eps_clip,
        has_continuous_action_space,
        action_std_init
    )
    agent.load(checkpoint_path)

    # 评估用确定性动作：最关键的一行
    if hasattr(agent, "set_action_std"):
        agent.set_action_std(1e-6) 

    returns = []
    lengths = []

    for ep in range(n_eval_episodes):
        s, _ = eval_env.reset(seed=seed + ep)
        done = False
        ep_ret = 0.0
        ep_len = 0

        _clear_buffer(agent)  # 每个 episode 清一次就够了

        while not done:
            a = agent.select_action(s)   # 现在因为 std=0，等价 deterministic
            s, r, terminated, truncated, _ = eval_env.step(a)
            done = terminated or truncated
            ep_ret += float(r)
            ep_len += 1

        _clear_buffer(agent)
        returns.append(ep_ret)
        lengths.append(ep_len)


    eval_env.close()

    arr = np.array(returns, dtype=np.float32)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "episodes": n_eval_episodes,
        "avg_len": float(np.mean(lengths)),
    }



# env_name = "reacher2d_2joints"
ENV_ID = "gymnasium_env/Reacher2D-v0"
XML_FILE = "./gymnasium_env/envs/test2.xml"
env_name = "Reacher-v5"
max_episode_steps = 100
has_continuous_action_space = True
env = gymnasium.make(ENV_ID, xml_file=XML_FILE)
env = TimeLimit(env, max_episode_steps = max_episode_steps)
state_dim = env.observation_space.shape[0]
print(f"initial state dim:{state_dim}")

# if has_continuous_action_space:
action_dim = env.action_space.shape[0]
# 
# print(state_dim)



random_seed = 0         # set random seed if required (0 = no random seed)
time_step = 0
i_episode = 0
max_ep_len = 1000
max_training_timesteps = int(3e6)
print_freq = max_ep_len * 10        # print avg reward in the interval (in num timesteps)
log_freq = max_ep_len * 2
save_model_freq = int(1e5)
K_epochs = 80               # update policy for K epochs in one PPO update

eps_clip = 0.2          # clip parameter for PPO
gamma = 0.99            # discount factor

lr_actor = 0.0003       # learning rate for actor network
lr_critic = 0.001       # learning rate for critic network

action_std = 0.6                    # starting std for action distribution (Multivariate Normal)
action_std_decay_rate = 0.05        # linearly decay action_std (action_std = action_std - action_std_decay_rate)
min_action_std = 0.1                # minimum action_std (stop decay after action_std <= min_action_std)
action_std_decay_freq = int(2.5e5)  # action_std decay frequency (in num timesteps)

update_timestep = max_ep_len * 4


#### log files for multiple runs are NOT overwritten
log_dir = "PPO_logs"
if not os.path.exists(log_dir):
        os.makedirs(log_dir)

log_dir = log_dir + '/' + env_name + '/'
if not os.path.exists(log_dir):
        os.makedirs(log_dir)


run_num = 0
current_num_files = next(os.walk(log_dir))[2]
run_num = len(current_num_files)

#### create new log file for each run
log_f_name = log_dir + '/PPO_' + env_name + "_log_" + str(run_num) + ".csv"

print("current logging run number for " + env_name + " : ", run_num)
print("logging at : " + log_f_name)
#####################################################

################### checkpointing ###################
run_num_pretrained = 0      #### change this to prevent overwriting weights in same env_name folder

directory = "PPO_preTrained"
if not os.path.exists(directory):
        os.makedirs(directory)

directory = directory + '/' + env_name + '/'
if not os.path.exists(directory):
        os.makedirs(directory)


checkpoint_path = directory + "PPO_{}_{}_{}.pth".format(env_name, random_seed, run_num_pretrained)
print("save checkpoint path : " + checkpoint_path)
#####################################################


############# print all hyperparameters #############
print("--------------------------------------------------------------------------------------------")
print("max training timesteps : ", max_training_timesteps)
print("max timesteps per episode : ", max_ep_len)
print("model saving frequency : " + str(save_model_freq) + " timesteps")
print("log frequency : " + str(log_freq) + " timesteps")
print("printing average reward over episodes in last : " + str(print_freq) + " timesteps")
print("--------------------------------------------------------------------------------------------")
print("state space dimension : ", state_dim)
print("action space dimension : ", action_dim)
print("--------------------------------------------------------------------------------------------")
if has_continuous_action_space:
    print("Initializing a continuous action space policy")
    print("--------------------------------------------------------------------------------------------")
    print("starting std of action distribution : ", action_std)
    print("decay rate of std of action distribution : ", action_std_decay_rate)
    print("minimum std of action distribution : ", min_action_std)
    print("decay frequency of std of action distribution : " + str(action_std_decay_freq) + " timesteps")
else:
    print("Initializing a discrete action space policy")
print("--------------------------------------------------------------------------------------------")
print("PPO update frequency : " + str(update_timestep) + " timesteps")
print("PPO K epochs : ", K_epochs)
print("PPO epsilon clip : ", eps_clip)
print("discount factor (gamma) : ", gamma)
print("--------------------------------------------------------------------------------------------")
print("optimizer learning rate actor : ", lr_actor)
print("optimizer learning rate critic : ", lr_critic)
if random_seed:
    print("--------------------------------------------------------------------------------------------")
    print("setting random seed to ", random_seed)
    torch.manual_seed(random_seed)
    env.seed(random_seed)
    np.random.seed(random_seed)
#####################################################

print("============================================================================================")

################# training procedure ################

# initialize a PPO agent
ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std)

# track total training time
start_time = datetime.now().replace(microsecond=0)
print("Started training at (GMT) : ", start_time)

print("============================================================================================")

# logging file
log_f = open(log_f_name,"w+")
log_f.write('episode,timestep,reward\n')

# printing and logging variables
print_running_reward = 0
print_running_episodes = 0

log_running_reward = 0
log_running_episodes = 0

time_step = 0
i_episode = 0


# wandb.init(
#     project="reacher2d-ppo",
#     name=f"{env_name}-seed{random_seed}-run{run_num}",
#     mode="offline",   # 关键：离线
#     config={
#         "env_name": env_name,
#         "max_episode_steps": max_episode_steps,
#         "max_ep_len": max_ep_len,
#         "max_training_timesteps": max_training_timesteps,
#         "update_timestep": update_timestep,
#         "K_epochs": K_epochs,
#         "eps_clip": eps_clip,
#         "gamma": gamma,
#         "lr_actor": lr_actor,
#         "lr_critic": lr_critic,
#         "action_std": action_std,
#         "action_std_decay_rate": action_std_decay_rate,
#         "min_action_std": min_action_std,
#         "action_std_decay_freq": action_std_decay_freq,
#         "random_seed": random_seed,
#     },
# )


tb = TBLogger(root="runs", exp_name=env_name, run_name=f"seed{random_seed}_run{run_num}")
tb.add_hparams({
    "max_training_timesteps": max_training_timesteps,
    "max_ep_len": max_ep_len,
    "update_timestep": update_timestep,
    "K_epochs": K_epochs,
    "eps_clip": eps_clip,
    "gamma": gamma,
    "lr_actor": lr_actor,
    "lr_critic": lr_critic,
    "action_std": action_std,
    "action_std_decay_rate": action_std_decay_rate,
    "min_action_std": min_action_std,
    "action_std_decay_freq": action_std_decay_freq,
    "random_seed": random_seed,
})
print("TensorBoard logdir:", tb.logdir)




while time_step <= max_training_timesteps:

    state, _ = env.reset()
    current_ep_reward = 0


    for t in range(1, max_ep_len + 1):
        action = ppo_agent.select_action(state)
        # print(action.shape)
        state, reward, terminated, truncated, info = env.step(action)

        ppo_agent.buffer.rewards.append(reward)
        done = terminated or truncated
        ppo_agent.buffer.is_terminals.append(done)

        time_step += 1
        current_ep_reward += reward


        if time_step % update_timestep == 0:
            ppo_agent.update()


        if has_continuous_action_space and time_step % action_std_decay_freq == 0:

            ppo_agent.decay_action_std(action_std_decay_freq, min_action_std)


            # log in logging file
        if time_step % log_freq == 0:

            # log average reward till last episode
            log_avg_reward = log_running_reward / log_running_episodes
            log_avg_reward = round(log_avg_reward, 4)

            log_f.write('{},{},{}\n'.format(i_episode, time_step, log_avg_reward))
            log_f.flush()

            log_running_reward = 0
            log_running_episodes = 0

        # printing average reward
        if time_step % print_freq == 0:

            # print average reward till last episode
            print_avg_reward = print_running_reward / print_running_episodes
            print_avg_reward = round(print_avg_reward, 2)

            print("Episode : {} \t\t Timestep : {} \t\t Average Reward : {}".format(i_episode, time_step, print_avg_reward))

            print_running_reward = 0
            print_running_episodes = 0

        # save model weights
            # save model weights + evaluate
        if time_step % save_model_freq == 0:
            print("--------------------------------------------------------------------------------------------")

            # 1) 先存一个带 step 的临时名字（eval 后再决定最终名字也行）
            tmp_ckpt = directory + f"PPO_{env_name}_seed{random_seed}_run{run_num}_step{time_step}.pth"
            print("saving model at : " + tmp_ckpt)
            ppo_agent.save(tmp_ckpt)
            print("model saved, now evaluating...")

            # 2) 立刻 load 这个 ckpt 并 eval（建议 20~100 集；为了和你 print 窗口 ~100 集对齐，可用 100）
            eval_stats = evaluate_checkpoint(
                env_name=env_name,
                checkpoint_path=tmp_ckpt,
                max_episode_steps=max_episode_steps,
                n_eval_episodes=50,          # 你可以改成 20/50/100
                seed=12345,                  # eval seed 固定，便于对比不同 checkpoint
                state_dim=state_dim,
                action_dim=action_dim,
                lr_actor=lr_actor,
                lr_critic=lr_critic,
                gamma=gamma,
                K_epochs=K_epochs,
                eps_clip=eps_clip,
                has_continuous_action_space=has_continuous_action_space,
                action_std_init=action_std,  # 注意：如果你的 load 会覆盖 std，这里只是占位
                render=True,
            )

            mean_r = eval_stats["mean"]
            std_r = eval_stats["std"]

            print(f"[EVAL] step={time_step}  mean={mean_r:.3f}  std={std_r:.3f}  "
                f"min={eval_stats['min']:.3f}  max={eval_stats['max']:.3f}  "
                f"avg_len={eval_stats['avg_len']:.1f}")

            # 3) 把 eval mean 写进文件名（用整数更方便排序：-5.12 -> -512）
            mean_tag = int(round(mean_r * 100))
            final_ckpt = directory + f"PPO_{env_name}_seed{random_seed}_run{run_num}_step{time_step}_EvalR{mean_tag}.pth"

            # 重命名（如果目标已存在就先删掉）
            if os.path.exists(final_ckpt):
                os.remove(final_ckpt)
            os.rename(tmp_ckpt, final_ckpt)

            print("renamed checkpoint to : " + final_ckpt)
            print("Elapsed Time  : ", datetime.now().replace(microsecond=0) - start_time)
            print("--------------------------------------------------------------------------------------------")


        # break; if the episode is over
        if done:
            break
    

    # wandb.log({
    #     "charts/episode_reward": current_ep_reward,
    #     "charts/episode_length": t,
    #     "charts/episode": i_episode,
    # },
    # step = time_step
    # )

    print_running_reward += current_ep_reward
    print_running_episodes += 1

    log_running_reward += current_ep_reward
    log_running_episodes += 1

    i_episode += 1

    tb.log_episode(
        episode_reward=current_ep_reward,
        episode_len=t,
        episode_idx=i_episode,
        global_step=time_step,
    )

log_f.close()     
env.close()
tb.close()




# print total training time
print("============================================================================================")
end_time = datetime.now().replace(microsecond=0)
print("Started training at (GMT) : ", start_time)
print("Finished training at (GMT) : ", end_time)
print("Total training time  : ", end_time - start_time)
print("============================================================================================")











