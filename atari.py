from stable_baselines3.common.env_util import make_atari_env
import torch
import numpy as np
from icm import ICM
from replay_buffer import ReplayBuffer
from sac import SACNetwork, sac_loss
import gymnasium as gym
import ale_py

gym.register_envs(ale_py)

def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    env = make_atari_env(f'ALE/{args.env}-v5', args.num_envs, args.seed)
    num_actions = env.action_space.n
    sac_network = SACNetwork(num_actions, args.ensemble_size, args.tau).to(device)
    optim = torch.optim.AdamW(sac_network.parameters(), args.lr)
    replay_buffer = ReplayBuffer(env.observation_space.shape, num_actions, env.observation_space.dtype, args.minibatch_size, args.replay_capacity, device)
    obs = env.reset()
    for i in range(args.timesteps):
        action = sac_network.policy_fn(obs)
        next_obs, reward, done, _ = env.step(action)
        replay_buffer.add_batch(obs, action, next_obs, reward, done)
        if i > args.train_after:
            if i % args.update_every == 0:
                for _ in range(args.num_epochs):
                    loss, metrics = sac_loss(sac_network, optim, replay_buffer, args.discount, args.alpha, args.critic_coef)

    
if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--env', type=str, default='Breakout') #MontezumaRevenge
    parser.add_argument('--discount', type=float, default=0.99)
    parser.add_argument('--alpha', type=float, default=0.99)
    parser.add_argument('--critic_coef', type=float, default=1)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--ensemble_size', type=int, default=2)
    parser.add_argument('--tau', type=float, default=1e-4)
    parser.add_argument('--minibatch_size', type=int, default=64)
    parser.add_argument('--replay_capacity', type=int, default=1_000_000)
    parser.add_argument('--num_envs', type=int, default=16)
    parser.add_argument('--update_every', type=int, default=4)
    parser.add_argument('--train_after', type=int, default=10_000)
    parser.add_argument('--timesteps', type=int, default=1_000_000)
    parser.add_argument('--num_epochs', type=int, default=1)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    main(args)
    