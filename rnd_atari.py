from stable_baselines3.common.env_util import make_atari_env, make_vec_env
from stable_baselines3.common.vec_env import VecNormalize, VecTransposeImage, VecFrameStack
import torch
from torch import optim
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
import ale_py
from rnd import RND, AtariFeatureModel, RunningMeanStd, AtariPolicyNetwork, AtariRNDValueNetwork, PPONetwork, train_rnd, collect_rollout
import imageio
from utils import get_action_dim, to_numpy
from logger import Logger

gym.register_envs(ale_py)

def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    logger = Logger('logs')
    
    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    env = make_atari_env(f'ALE/{args.env}-v5', args.num_envs, args.seed, wrapper_kwargs={'terminal_on_life_loss': False})
    env = VecTransposeImage(VecFrameStack(env, 4))
    eval_env = make_atari_env(f'ALE/{args.env}-v5', 1, args.seed, wrapper_kwargs={'terminal_on_life_loss': False})
    eval_env = VecTransposeImage(VecFrameStack(eval_env, 4))
    
    def eval(ppo_network, eval_env):
        done = False
        obs = eval_env.reset()
        total_rew = 0
        frames = []
        while not done:
            obs = torch.as_tensor((obs - rms.mean) / (np.sqrt(rms.var) + 1e-8)).to(device).float()
            action, _ = ppo_network.policy_network.policy_fn(obs, det=True)
            obs, reward, done, _ = eval_env.step(to_numpy(action))
            frames.append(eval_env.render())
            total_rew += reward
        return total_rew, frames

    with torch.device(device):
        feature_model = AtariFeatureModel()
        target_model = AtariFeatureModel()
        rnd_opt = optim.Adam(feature_model.parameters(), lr=args.lr)
        rnd = RND(feature_model, target_model)
        
        policy = AtariPolicyNetwork(env.action_space.n)
        value = AtariRNDValueNetwork()
        ppo_network = PPONetwork(policy, value)
        ppo_opt = optim.Adam(ppo_network.parameters(), lr=args.lr)
    
        rms = RunningMeanStd()
        obs = env.reset()
        rms.update(obs)
        obs = (obs - rms.mean) / (np.sqrt(rms.var) + 1e-8)
        
        for i in range(args.timesteps // (args.num_envs * args.rollout_length)):
            rollout, obs = collect_rollout(env, ppo_network, args.rollout_length, obs, rms)
            metrics = train_rnd(rollout, ppo_network, rnd, ppo_opt, rnd_opt, args.minibatch_size, args.num_epochs, device, 
                                max_grad_norm=args.max_grad_norm, rnd_reward_coef=0, ent_coef=1e-3)
            logger.add_metrics(metrics)
            train_rew = np.sum(rollout.rewards) / max(1, np.sum(rollout.dones))
            print(train_rew)
            logger.add_scalar('train_reward', train_rew)
            if i % 10 == 0:
                eval_reward, frames = eval(ppo_network, eval_env)
                print(f'epoch {i}: {np.mean(eval_reward)}')
                logger.add_scalar('eval_reward', eval_reward)
                imageio.mimwrite(f'gifs/{args.env}_{i}.gif', frames, loop=0, fps=20)
            logger.write(i * args.num_envs * args.rollout_length)
    
    
if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--env', type=str, default='Breakout') #MontezumaRevenge
    parser.add_argument('--discount', type=float, default=0.99)
    parser.add_argument('--critic_coef', type=float, default=1)
    parser.add_argument('--max_grad_norm', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--rollout_length', type=int, default=128)
    parser.add_argument('--num_envs', type=int, default=64)
    parser.add_argument('--timesteps', type=int, default=2_000_000)
    parser.add_argument('--log_every', type=int, default=10)
    parser.add_argument('--num_epochs', type=int, default=4)
    parser.add_argument('--minibatch_size', type=int, default=256)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    main(args)
    