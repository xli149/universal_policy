import os
import numpy as np
import torch
import gymnasium as gym
from gymnasium.wrappers import TimeLimit

from ppo import PPO


def evaluate(
    env_name="Reacher-v5",
    checkpoint_path=None,
    max_episode_steps=100,
    has_continuous_action_space=True,
    action_std=0.6,
    n_eval_episodes=20,
    seed=0,
    render=False,
):
    # ---------- check ----------
    assert checkpoint_path is not None, "Please provide checkpoint_path"
    assert os.path.exists(checkpoint_path), f"checkpoint not found: {checkpoint_path}"

    # ---------- env ----------
    render_mode = "human" if render else None
    env = gym.make(env_name, render_mode=render_mode)
    env = TimeLimit(env, max_episode_steps=max_episode_steps)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # ---------- agent (IMPORTANT: same signature as your training code) ----------
    ppo_agent = PPO(
        state_dim,
        action_dim,
        3e-4,   # lr_actor (dummy for eval)
        1e-3,   # lr_critic (dummy for eval)
        0.99,   # gamma (dummy for eval)
        80,     # K_epochs (dummy for eval)
        0.2,    # eps_clip (dummy for eval)
        has_continuous_action_space,
        action_std,
    )

    # load weights
    ppo_agent.load(checkpoint_path)

    # seed for reproducibility
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    episode_returns = []
    episode_lens = []

    # ---------- rollouts ----------
    for ep in range(n_eval_episodes):
        ep_seed = None if seed is None else (seed + ep)
        state, _ = env.reset(seed=ep_seed)

        done = False
        ep_return = 0.0
        ep_len = 0

        while not done:
            # use your existing action selection
            action = ppo_agent.select_action(state)

            # very important: in eval we should NOT store into buffer
            # Some PPO implementations push to buffer inside select_action.
            # So we clear it each step to avoid memory growth / interference.
            if hasattr(ppo_agent, "buffer"):
                if hasattr(ppo_agent.buffer, "states"):
                    ppo_agent.buffer.states.clear()
                if hasattr(ppo_agent.buffer, "actions"):
                    ppo_agent.buffer.actions.clear()
                if hasattr(ppo_agent.buffer, "logprobs"):
                    ppo_agent.buffer.logprobs.clear()
                if hasattr(ppo_agent.buffer, "rewards"):
                    ppo_agent.buffer.rewards.clear()
                if hasattr(ppo_agent.buffer, "is_terminals"):
                    ppo_agent.buffer.is_terminals.clear()

            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            ep_return += float(reward)
            ep_len += 1

        episode_returns.append(ep_return)
        episode_lens.append(ep_len)
        print(f"[Eval] Episode {ep+1:03d}/{n_eval_episodes} | return={ep_return:.3f} | len={ep_len}")

    env.close()

    rets = np.array(episode_returns, dtype=np.float32)
    lens = np.array(episode_lens, dtype=np.int32)

    print("\n================ Evaluation Summary ================")
    print(f"Env: {env_name}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Episodes: {n_eval_episodes}")
    print(f"Return: mean={rets.mean():.3f}, std={rets.std():.3f}, min={rets.min():.3f}, max={rets.max():.3f}")
    print(f"Length: mean={lens.mean():.1f}, std={lens.std():.1f}, min={lens.min()}, max={lens.max()}")
    print("===================================================\n")

    return rets, lens


if __name__ == "__main__":
    env_name = "Reacher-v5"
    random_seed = 0
    run_num_pretrained = 0

    checkpoint_path = f"PPO_preTrained/Reacher-v5/PPO_Reacher-v5_0_0.pth"

    evaluate(
        env_name=env_name,
        checkpoint_path=checkpoint_path,
        max_episode_steps=100,
        n_eval_episodes=20,
        seed=123,
        render=True,
    )
