import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Categorical, TanhTransform
import copy
import numpy as np

from models import MLP, AtariConv
from utils import to_numpy, get_action_dim, get_obs_shape

class Rollout:
    obs: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    log_probs: np.ndarray
    last_obs: np.ndarray
    
    def __init__(self, rollout_length, num_envs, obs_space, action_space):
        self.rollout_length = rollout_length
        self.num_envs = num_envs
        self.obs_shape = get_obs_shape(obs_space)
        self.action_dim = get_action_dim(action_space)
        self.ind = 0
        self.reset()
    
    def reset(self):
        self.obs = np.zeros((self.rollout_length, self.num_envs, *self.obs_shape), dtype=np.float32)
        self.actions = np.zeros((self.rollout_length, self.num_envs, self.action_dim), dtype=np.float32)
        self.rewards = np.zeros((self.rollout_length, self.num_envs), dtype=np.float32)
        self.dones = np.zeros((self.rollout_length, self.num_envs), dtype=np.float32)
        self.log_probs = np.zeros((self.rollout_length, self.num_envs), dtype=np.float32)
    
    def add(self, obs, action, reward, done, log_prob):
        self.obs[self.ind] = obs
        self.actions[self.ind] = action
        self.rewards[self.ind] = reward
        self.dones[self.ind] = done
        self.log_probs[self.ind] = log_prob
        self.ind += 1
        
    def unpack(self):
        return self.obs, self.actions, self.rewards, self.dones, self.log_probs, self.last_obs

class AtariPolicyNetwork(nn.Module):
    def __init__(self, num_actions):
        super(AtariPolicyNetwork, self).__init__()
        self.conv = AtariConv()
        self.mlp = MLP(self.conv.output_dim, num_actions)
    
    def forward(self, x):
        x = self.conv(x)
        x = x.reshape(-1, self.conv.output_dim)
        x = self.mlp(x)
        return x
    
    def policy_dist(self, x):
        logits = self(x)
        action_dist = Categorical(logits=logits)
        return action_dist
    
    def policy_fn(self, x):
        # actually choosing the action
        action_dist = self.policy_dist(x)
        action = action_dist.sample()
        return action, action_dist.log_prob(action) # [B, 1]

class AtariValueNetwork(nn.Module):
    def __init__(self, num_actions):
        super(AtariValueNetwork, self).__init__()
        self.conv = AtariConv()
        self.mlp = MLP(self.conv.output_dim, num_actions)
    
    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, self.conv.output_dim)
        x = self.mlp(x)
        return x # [B, 1]

class PPONetwork(nn.Module):
    def __init__(self, policy_network, value_network):
        super(PPONetwork, self).__init__()
        self.policy_network = policy_network
        self.value_network = value_network

def compute_gae(rewards, values, dones, discount=0.99, lam=0.95):
    # from here https://github.com/zplizzi/pytorch-ppo/blob/master/gae.py
    # values is T+1, since it needs to include the value for the next_obs, also purely numpy
    N = rewards.shape[1]
    T = rewards.shape[0]
    gae_step = np.zeros((N, ), dtype=np.float32)
    advantages = np.zeros((T, N), dtype=np.float32)
    for t in reversed(range(T)):
        delta = rewards[t] + discount * values[t + 1] * (1 - dones[t]) - values[t]
        gae_step = delta + discount * lam * (1 - dones[t]) * gae_step
        advantages[t] = gae_step
    returns = advantages + values[:-1]
    return advantages, returns

def compute_ppo_loss(states, rewards, dones, discount=0.99, lam=0.95):
    pass