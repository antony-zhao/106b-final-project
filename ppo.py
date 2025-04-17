import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Categorical, TanhTransform
import copy
import numpy as np

from models import MLP, AtariConv

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
        action_dist = self.action_dist(x)
        action = action_dist.sample()
        return action # [B, 1]

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
    N = rewards.shape[0]
    T = rewards.shape[1]
    gae_step = np.zeros((N, ))
    advantages = np.zeros((N, T))
    for t in reversed(range(T - 1)):
        delta = rewards[:, t] + discount * values[:, t + 1] * (1 - dones[:, t]) - values[:, t]
        gae_step = delta + discount * lam * (1 - dones[:, t]) * gae_step
        advantages[:, t] = gae_step
    return advantages

def compute_ppo_loss(states, rewards, dones, discount=0.99, lam=0.95):
    pass