import numpy as np
import gymnasium as gym
import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from rl.rnd import RND, RunningMeanStd, PPONetwork, train_rnd, RND_Rollout, layer_init
from rl.utils import to_numpy
from rl.models import ResBlock, MLP

class RobosuiteCore(nn.Module):
    def __init__(self, hidden=64, camera_dim=64, framestack=4, proprio_dim=37, act=nn.SELU, num_hidden=3):
        super().__init__()
        self.act = act()
        self.proprio_dim = proprio_dim
        self.hidden = hidden
        self.convs = nn.Sequential(
            ResBlock(in_channels=framestack, out_channels=16, kernel_size=7, stride=4),
            act(),
            ResBlock(in_channels=16, out_channels=32, kernel_size=5, stride=2),
            act(),
            ResBlock(in_channels=32, out_channels=32, kernel_size=3, stride=2),
            act(),
            ResBlock(in_channels=32, out_channels=64, kernel_size=3, stride=2),
            act()
        )
        self.output_dim = self.compute_output_dim(framestack, camera_dim)
        self.proprio_mlp = nn.Linear(proprio_dim * framestack, hidden)
        self.mlp = MLP(self.output_dim + 64, hidden, hidden_dim=hidden, act=act, num_hiddens=num_hidden)
        self.image_inds = list(range(0, framestack * 2, 2))
        self.proprio_inds = list(range(1, framestack * 2, 2))
    
    def forward(self, x):
        '''
        x has B C H W, and C[0, 2, 4, 6] are images, and C[1, 3, 5, 7] are proprio states
        '''
        dim = len(x.shape)
        if dim == 3:
            images = x[self.image_inds]
            proprio = x[self.proprio_inds] # B, 4, H, W
            proprio = proprio.flatten(1)[:, :self.proprio_dim]
            proprio = proprio.flatten()
            conv_out = self.convs(images).flatten()
        else:
            images = x[:, self.image_inds]
            proprio = x[:, self.proprio_inds] # B, 4, H, W
            proprio = proprio.flatten(2)[:, :, :self.proprio_dim]
            proprio = proprio.flatten(1)
            conv_out = self.convs(images).flatten(1)
        proprio_out = self.act(self.proprio_mlp(proprio))
        x = torch.concat([conv_out, proprio_out], dim=-1)
        out = self.act(self.mlp(x))
        return out

    def compute_output_dim(self, input_channels, camera_dim):
        x = torch.zeros(1, input_channels, camera_dim, camera_dim)
        x = self.convs(x)
        return x.view(-1).shape[0]

class RobosuitePolicy(nn.Module):
    def __init__(self, n_actions=7, camera_dim=64, framestack=4, act=nn.SELU):
        super().__init__()
        self.core = RobosuiteCore(camera_dim=camera_dim, framestack=framestack, act=act)
        self.action_mean = nn.Linear(self.core.hidden, n_actions)
        self.log_std = nn.Parameter(torch.zeros(n_actions))
    
    def forward(self, x):
        x = self.core(x)
        return self.action_mean(x)
    
    def policy_dist(self, x):
        logits = self(x)
        action_dist = Normal(loc=logits, scale=torch.exp(self.log_std))
        return action_dist
    
    def policy_fn(self, x, det=False):
        # actually choosing the action
        if not det:
            action_dist = self.policy_dist(x)
        if det:
            action_mean = self(x)
            return action_mean, None
        action = action_dist.rsample()
        return action, action_dist.log_prob(action).sum(-1)

class RobosuiteValue(nn.Module):
    def __init__(self, camera_dim=64, framestack=4, act=nn.SELU):
        super().__init__()
        self.core = RobosuiteCore(camera_dim=camera_dim, framestack=framestack, act=act)
        self.value = nn.Linear(self.core.hidden, 1)
        self.value_rnd = nn.Linear(self.core.hidden, 1)
    
    def forward(self, x):
        x = self.core(x)
        return self.value(x), self.value_rnd(x)
    
class RobosuiteFeatureModel(nn.Module):
    def __init__(self, camera_dim=64, framestack=1, num_hidden=3, feature_dim=512, act=nn.SELU):
        super().__init__()
        self.core = RobosuiteCore(camera_dim=camera_dim, framestack=framestack, act=act, num_hidden=num_hidden)
        self.feature = nn.Linear(self.core.hidden, feature_dim)
    
    def forward(self, x):
        x = self.core(x)
        return self.feature(x)
    
def collect_rollout(env, ppo_network, rnd, rollout_length, obs, rms):
    rollout = RND_Rollout(rollout_length, env.num_envs, env.observation_space, env.action_space)
    total_reward = 0
    num_completed = 0
    for _ in range(rollout_length):
        obs_tensor = torch.as_tensor(obs).float()
        action, log_prob = ppo_network.policy_network.policy_fn(obs_tensor)
        action = to_numpy(action)
        tanh_action = np.tanh(action)
        # tanh_action[:, 0] = 0
        log_prob = to_numpy(log_prob)
        next_obs, reward, done, infos = env.step(tanh_action)
        intrinsic_reward, _ = rnd.compute_intrinsic(torch.as_tensor((next_obs - rms.mean) / (np.sqrt(rms.var) + 1e-8)).float())
        next_obs = next_obs
        for info in infos:
            if 'episode' in info.keys():
                total_reward += info['episode']['r']
                num_completed += 1
        rollout.add(obs, action, reward, intrinsic_reward, done, log_prob)
        obs = next_obs
    rollout.last_obs = obs
    return rollout, obs, total_reward / num_completed if num_completed > 0 else None
