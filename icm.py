import torch
from torch import nn
import torch.nn.functional as F
import gymnasium as gym
from stable_baselines3.common.env_util import make_atari_env
from torch.distributions import Categorical
import numpy as np

from models import MLP, AtariConv

class ICM(nn.Module):
    def __init__(self, feature_model, forward_model, inverse_model, is_discrete=False):
        super(ICM, self).__init__()
        self.feature_model = feature_model
        self.forward_model = forward_model
        self.inverse_model = inverse_model
        self.is_discrete = is_discrete
    
    def compute_forward_loss(self, obs, next_obs, action):
        feature = self.feature_model(obs)
        next_feature = self.feature_model(next_obs)
        predicted_next_feature = self.forward_model(feature, action)
        
        error = 0.5 * (next_feature - predicted_next_feature) ** 2
        return torch.mean(error), error

    def compute_inverse_loss(self, obs, next_obs, action):
        feature = self.feature_model(obs)
        next_feature = self.feature_model(next_obs)
        predicted_action = self.inverse_model(feature, next_feature)
        
        if self.is_discrete:
            error = F.cross_entropy(predicted_action, action)
        else:
            error = F.mse_loss(predicted_action, action)
        return error
        
    
    def compute_intrinsic(self, obs, next_obs, action):
        forward_loss, error = self.compute_forward_loss(obs, next_obs, action)
        inverse_loss = self.compute_inverse_loss(obs, next_obs, action)

        loss = forward_loss + inverse_loss
        return error.mean(-1).detach().cpu().numpy(), loss

class InverseModel(nn.Module):
    def __init__(self, feature_dim, action_dim):
        super(InverseModel, self).__init__()
        self.mlp = MLP(feature_dim * 2, action_dim)
    
    def forward(self, feature1, feature2):
        x = torch.concat([feature1, feature2], dim=1)
        return self.mlp(x)
    
class ForwardModel(nn.Module):
    def __init__(self, feature_dim, action_dim):
        super(ForwardModel, self).__init__()
        self.mlp = MLP(feature_dim + action_dim, feature_dim)
    
    def forward(self, feature, action):
        x = torch.concat([feature, action], dim=1)
        return self.mlp(x)
    