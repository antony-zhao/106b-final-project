from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env.vec_frame_stack import VecFrameStack
from stable_baselines3.common.vec_env.vec_transpose import VecTransposeImage
from stable_baselines3.common.vec_env.vec_monitor import VecMonitor
from  stable_baselines3.common.monitor import Monitor
import torch
import numpy as np
from icm import ICM
from replay_buffer import ReplayBuffer
from sac import SACNetwork, sac_loss, discrete_sac_loss, AtariActorNetwork, AtariCriticNetwork
import gymnasium as gym
import ale_py
from logger import Logger
from collections import deque
from torch import nn

gym.register_envs(ale_py)

def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    logger = Logger('logs')
    
    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    env = make_atari_env(f'ALE/{args.env}-v5', args.num_envs, args.seed)
    env = VecTransposeImage(VecFrameStack(env, 4))
    
    action_dim = env.action_space.shape
    num_actions = env.action_space.n
    actor_network = AtariActorNetwork(num_actions)
    critic_networks = nn.ModuleList([AtariCriticNetwork(num_actions) for _ in range(args.ensemble_size)])
    actor_optim = torch.optim.Adam(actor_network.parameters(), args.lr)
    critic_optim = torch.optim.Adam(critic_networks.parameters(), args.lr)
    sac_network = SACNetwork(actor_network, critic_networks, args.tau).to(device)
    replay_buffer = ReplayBuffer(env.observation_space.shape, action_dim, env.observation_space.dtype, args.minibatch_size, args.replay_capacity, device)
    
    rewards = deque(maxlen=args.log_every)
    dones = deque(maxlen=args.log_every)
    
    obs = env.reset()
    for _ in range(args.init_steps):
        obs, _, _, _ = env.step([env.action_space.sample() for _ in range(args.num_envs)])
    for i in range(args.timesteps):
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device)
        action = sac_network.act(obs_tensor).cpu().numpy()
        next_obs, reward, done, _ = env.step(action)
        rewards.append(reward)
        dones.append(done)
        replay_buffer.add_batch(obs, action, reward, next_obs, done)
        obs = next_obs
        if i * args.num_envs > args.train_after:
            if i % args.update_every == 0:
                for _ in range(args.num_epochs):
                    loss, metrics = discrete_sac_loss(sac_network, actor_optim, critic_optim, replay_buffer, args.discount, args.alpha, args.critic_coef, args.max_grad_norm)
                logger.add_metrics(metrics)
                logger.add_scalar("loss", loss)
                logger.write(i)
        if (i + 1) % args.log_every == 0:
            average_episode_reward = np.sum(rewards) / max(np.sum(dones), 1)
            logger.add_scalar("reward", average_episode_reward)
            logger.write(i)
            print(average_episode_reward)
    
if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--env', type=str, default='Breakout') #MontezumaRevenge
    parser.add_argument('--discount', type=float, default=0.99)
    parser.add_argument('--alpha', type=float, default=0.4)
    parser.add_argument('--critic_coef', type=float, default=1)
    parser.add_argument('--max_grad_norm', type=float, default=0.2)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--ensemble_size', type=int, default=2)
    parser.add_argument('--tau', type=float, default=1e-3)
    parser.add_argument('--minibatch_size', type=int, default=256)
    parser.add_argument('--replay_capacity', type=int, default=1_000_000)
    parser.add_argument('--num_envs', type=int, default=32)
    parser.add_argument('--update_every', type=int, default=1)
    parser.add_argument('--train_after', type=int, default=10_000)
    parser.add_argument('--timesteps', type=int, default=1_000_000)
    parser.add_argument('--log_every', type=int, default=128)
    parser.add_argument('--num_epochs', type=int, default=1)
    parser.add_argument('--init_steps', type=int, default=50)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    main(args)
    