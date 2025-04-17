import torch
from torch import nn
import torch.nn.functional as F
import gymnasium as gym
from stable_baselines3.common.env_util import make_atari_env
from torch.distributions import Categorical
import numpy as np

from models import MLP, AtariConv

class RND(nn.Module):
    def __init__(self, feature_model, target_model):
        super(RND, self).__init__()
        self.feature_model = feature_model
        self.feature_model_target = target_model
        self.rms = RunningMeanStd()
    
    def compute_intrinsic(self, next_obs):
        features = self.feature_model(next_obs)
        with torch.no_grad():
            target_features = self.feature_model_target(next_obs)

        error = 0.5 * (features - target_features) ** 2
        intrinsic_reward = error.mean(-1).detach().cpu().numpy()
        self.rms.update(intrinsic_reward)
        intrinsic_reward = (intrinsic_reward) / (np.sqrt(self.rms.var) + 1e-8)
        return intrinsic_reward, error.mean()
    
class AtariFeatureModel(nn.Module):
    def __init__(self, feature_dim=256):
        super(AtariFeatureModel, self).__init__()
        self.conv = AtariConv()
        self.mlp = MLP(self.conv.output_dim, feature_dim)
    
    def forward(self, obs):
        x = self.conv(obs)
        x = x.reshape(-1, self.conv.output_dim)
        return self.mlp(x)
    
class RunningMeanStd(object):
    # https://github.com/openai/random-network-distillation/blob/f75c0f1efa473d5109d487062fd8ed49ddce6634/mpi_util.py#L179
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-4, shape=(), comm=None, use_mpi=True):
        self.mean = np.zeros(shape, 'float32')
        self.var = np.ones(shape, 'float32')
        self.count = epsilon
        self.comm = comm


    def update(self, x):
        batch_mean, batch_std, batch_count = np.mean(x, axis=0), np.std(x, axis=0), x.shape[0]
        batch_var = np.square(batch_std)
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * (self.count)
        m_b = batch_var * (batch_count)
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / (self.count + batch_count)
        new_var = M2 / (self.count + batch_count)

        new_count = batch_count + self.count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count
