from stable_baselines3.common.env_util import make_atari_env, make_vec_env
from stable_baselines3.common.vec_env import VecNormalize, VecTransposeImage, VecFrameStack
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CallbackList
from stable_baselines3 import PPO, SAC
import torch
from torch import optim
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
import ale_py
from icm import ICM, ForwardModel, InverseModel, AtariConv
from rnd import RND, AtariFeatureModel, RunningMeanStd
import imageio

gym.register_envs(ale_py)

class ICMCallback(BaseCallback):
    # Based on https://github.com/RLE-Foundation/RLeXplore/blob/main/2%20rlexplore_with_sb3.ipynb, but with our own icm code
    def __init__(self, icm, icm_optim, verbose=0):
        super(ICMCallback, self).__init__(verbose)
        self.icm = icm
        self.icm_optim = icm_optim
        self.buffer = None

    def init_callback(self, model: BaseAlgorithm) -> None:
        super().init_callback(model)
        self.buffer = self.model.rollout_buffer

    def _on_rollout_end(self):
        obs = torch.as_tensor(self.buffer.observations)
        next_obs = obs.clone()
        next_obs[:-1] = obs[1:]
        next_obs[-1] = torch.as_tensor(self.locals["new_obs"])
        actions = torch.as_tensor(self.buffer.actions).long()
        # rewards = torch.as_tensor(self.buffer.rewards)
        # dones = torch.as_tensor(self.buffer.episode_starts)
        obs = obs.view(-1, 4, 84, 84)
        next_obs = next_obs.view(-1, 4, 84, 84)
        actions = actions.view(-1, 1)
        actions = F.one_hot(actions).squeeze(1).float()
        intrinsic_rewards, loss = self.icm.compute_intrinsic(obs, next_obs, actions)
        self.icm_optim.zero_grad()
        loss.backward()
        self.icm_optim.step()
        # add the intrinsic rewards to the buffer (?)
        intrinsic_rewards = intrinsic_rewards.reshape(self.buffer.rewards.shape)
        self.buffer.rewards += intrinsic_rewards
    
    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: If the callback returns False, training is aborted early.
        """
        return True

class RNDCallback(BaseCallback):
    # Based on https://github.com/RLE-Foundation/RLeXplore/blob/main/2%20rlexplore_with_sb3.ipynb, but with our own icm code
    def __init__(self, rnd, opt, eval_env, verbose=0):
        super(RNDCallback, self).__init__(verbose)
        self.rnd = rnd
        self.opt = opt
        self.buffer = None
        self.rms = RunningMeanStd()
        self.eval_env = eval_env
        self.i = 0

    def init_callback(self, model: BaseAlgorithm) -> None:
        super().init_callback(model)
        self.buffer = self.model.rollout_buffer

    def _on_rollout_end(self):
        obs = self.buffer.observations
        next_obs = obs.copy()
        next_obs[:-1] = obs[1:]
        next_obs[-1] = self.locals["new_obs"]
        next_obs = next_obs.reshape(-1, 4, 84, 84)
        next_obs = (next_obs - self.rms.mean) / (np.sqrt(self.rms.var) + 1e-8)
        next_obs = torch.as_tensor(next_obs).float()
        intrinsic_rewards, loss = self.rnd.compute_intrinsic(next_obs)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        if (self.i % 10) == 0:
            obs = self.eval_env.reset()
            frames = []
            done = False
            while not done:
                action, _states = self.model.predict(obs)
                obs, rewards, done, info = self.eval_env.step(action)
                frames.append(self.eval_env.render())
            
            imageio.mimwrite(f'gifs/{args.env}_{self.i}.gif', frames)
        self.i += 1
        # add the intrinsic rewards to the buffer (?)
        intrinsic_rewards = intrinsic_rewards.reshape(self.buffer.rewards.shape)
        # self.buffer.rewards += intrinsic_rewards
    
    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: If the callback returns False, training is aborted early.
        """
        next_obs = self.locals["new_obs"].reshape(-1, 4, 84, 84)
        self.rms.update(next_obs)
        next_obs = (next_obs - self.rms.mean) / (np.sqrt(self.rms.var) + 1e-8)
        next_obs = np.clip(next_obs, -5, 5)
        next_obs = torch.as_tensor(next_obs).float()
        intrinsic_rewards, _ = self.rnd.compute_intrinsic(next_obs)        
        self.locals["rewards"] += intrinsic_rewards
        return True
        
def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    env = make_atari_env(f'ALE/{args.env}-v5', args.num_envs, args.seed)
    env = VecTransposeImage(VecFrameStack(env, 4))
    eval_env = make_atari_env(f'ALE/{args.env}-v5', 1, args.seed)
    eval_env = VecTransposeImage(VecFrameStack(eval_env, 4))

    feature_model = AtariFeatureModel()
    target_model = AtariFeatureModel()
    opt = optim.Adam(feature_model.parameters(), lr=args.lr)
    rnd = RND(feature_model, target_model)
    
    eval_callback = EvalCallback(eval_env, 
                             log_path=f"logs/{args.env}", eval_freq=500,
                             deterministic=False, render=True)
    
    model = PPO('CnnPolicy', env, verbose=1, device=device, n_steps=128, 
                tensorboard_log=f'logs/{args.env}', ent_coef=1e-3)
    model.learn(total_timesteps=args.timesteps * args.num_envs, callback=eval_callback)
    model.save(f'{env}_ppo')
    
    
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
    parser.add_argument('--num_envs', type=int, default=128)
    parser.add_argument('--update_every', type=int, default=1)
    parser.add_argument('--train_after', type=int, default=10_000)
    parser.add_argument('--timesteps', type=int, default=2_000_000)
    parser.add_argument('--log_every', type=int, default=128)
    parser.add_argument('--num_epochs', type=int, default=1)
    parser.add_argument('--init_steps', type=int, default=50)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    main(args)
    