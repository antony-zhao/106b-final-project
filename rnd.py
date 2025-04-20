import torch
from torch import nn
import torch.nn.functional as F
import gymnasium as gym
from stable_baselines3.common.env_util import make_atari_env
from torch.distributions import Categorical
import numpy as np

from models import MLP, AtariConv
from ppo import compute_gae, AtariPolicyNetwork, Rollout, PPONetwork
from utils import to_numpy

class RND(nn.Module):
    def __init__(self, feature_model, target_model):
        super(RND, self).__init__()
        self.feature_model = feature_model
        self.feature_model_target = target_model
        self.rms = RunningMeanStd()
    
    def compute_intrinsic(self, next_obs):
        next_obs = next_obs[:, [-1]]
        features = self.feature_model(next_obs)
        with torch.no_grad():
            target_features = self.feature_model_target(next_obs)

        error = 0.5 * (features - target_features) ** 2
        intrinsic_reward = to_numpy(error.mean(-1))
        self.rms.update(intrinsic_reward)
        intrinsic_reward = (intrinsic_reward) / (np.sqrt(self.rms.var) + 1e-8)
        return intrinsic_reward, error.mean()
    
class AtariFeatureModel(nn.Module):
    def __init__(self, feature_dim=256):
        super(AtariFeatureModel, self).__init__()
        self.conv = AtariConv(input_channels=1)
        self.mlp = MLP(self.conv.output_dim, feature_dim)
    
    def forward(self, obs):
        x = self.conv(obs)
        x = x.reshape(-1, self.conv.output_dim)
        return self.mlp(x)
    
class RunningMeanStd(object):
    # https://github.com/openai/random-network-distillation/blob/f75c0f1efa473d5109d487062fd8ed49ddce6634/mpi_util.py#L179
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, 'float32')
        self.var = np.ones(shape, 'float32')
        self.count = epsilon


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

class AtariRNDValueNetwork(nn.Module):
    def __init__(self, act=nn.SiLU):
        super(AtariRNDValueNetwork, self).__init__()
        self.conv = AtariConv()
        self.mlp = MLP(self.conv.output_dim, 256)
        self.val1 = nn.Linear(256, 1)
        self.val2 = nn.Linear(256, 1)
        self.act = act()
    
    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, self.conv.output_dim)
        x = self.act(self.mlp(x))
        v1 = self.val1(x)
        v2 = self.val2(x)
        return v1, v2

def compute_rnd_loss(ppo_network: PPONetwork, obs, actions, old_log_prob, reg_ret, rnd_ret, adv,
                     clip=0.2, ent_coef=0, val_coef=1):
    # compute_gae is numpy only
    values, rnd_values = ppo_network.value_network(obs)
    val_loss = F.mse_loss(reg_ret, values.squeeze(1)) + F.mse_loss(rnd_ret, rnd_values.squeeze(1))
    
    action_dist = ppo_network.policy_network.policy_dist(obs)
    action_log_prob = action_dist.log_prob(actions.squeeze(1))
    ratio = torch.exp(action_log_prob - old_log_prob)
    policy_loss1 = adv * ratio
    policy_loss2 = adv * torch.clamp(ratio, 1 - clip, 1 + clip)
    policy_loss = -torch.min(policy_loss1, policy_loss2).mean()
    
    ent_loss = -action_dist.entropy().mean()
    
    loss = ent_loss * ent_coef + policy_loss + val_loss * val_coef
    return loss, {'policy_loss': policy_loss, 'entropy_loss': ent_loss, 'value_loss': val_loss}

def collect_rollout(env, ppo_network, rollout_length, obs, rms):
    rollout = Rollout(rollout_length, env.num_envs, env.observation_space, env.action_space)
    for _ in range(rollout_length):
        obs_tensor = torch.as_tensor(obs).float()
        action, log_prob = ppo_network.policy_network.policy_fn(obs_tensor)
        action = to_numpy(action)
        log_prob = to_numpy(log_prob)
        next_obs, reward, done, _ = env.step(action)
        action = action[:, np.newaxis]
        rollout.add(obs, action, reward, done, log_prob)
        obs = next_obs
        rms.update(obs)
        obs = (obs - rms.mean) / (np.sqrt(rms.var) + 1e-8)
    rollout.last_obs = obs
    return rollout, obs

def train_rnd(rollout, ppo_network, rnd, ppo_optim, rnd_optim, minibatch_size, num_epochs, device, max_grad_norm=0.5,
              clip=0.2, discount=0.99, rnd_discount=0.999, lam=0.95, rnd_reward_coef=1, ent_coef=1e-3, val_coef=1):
    with torch.device(device):
        obs, actions, rewards, dones, log_probs, last_obs = rollout.unpack()
        last_obs = np.expand_dims(last_obs, 0)
        all_obs = torch.as_tensor(np.concatenate([obs, last_obs])).float()
        next_obs = all_obs[1:].reshape(-1, *all_obs.shape[2:])
        intrinsic_rewards, _ = rnd.compute_intrinsic(next_obs) #int_rewards are numpy
        intrinsic_rewards = intrinsic_rewards.reshape(rewards.shape) * rnd_reward_coef
        all_obs = all_obs.reshape(-1, *all_obs.shape[2:])
        values, rnd_values = ppo_network.value_network(all_obs)
        
        val_np = to_numpy(values).reshape(-1, rewards.shape[1])
        rnd_val_np = to_numpy(rnd_values).reshape(-1, rewards.shape[1])
        reg_adv, reg_ret = compute_gae(rewards, val_np, dones, discount, lam)
        rnd_adv, rnd_ret = compute_gae(intrinsic_rewards, rnd_val_np, dones, rnd_discount, lam)
        adv = reg_adv# + rnd_adv
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        
        num_samples = obs.shape[0] * obs.shape[1]
        obs = obs.reshape(num_samples, *obs.shape[2:])
        actions = actions.reshape(num_samples, *actions.shape[2:])
        log_probs = log_probs.reshape(num_samples, *log_probs.shape[2:])
        reg_ret = reg_ret.reshape(num_samples, *reg_ret.shape[2:])
        rnd_ret = rnd_ret.reshape(num_samples, *rnd_ret.shape[2:])
        adv = adv.reshape(num_samples, *adv.shape[2:])
        indices = np.random.permutation(num_samples)
        
        metrics = {'intrinsic_loss': 0}
        for _ in range(num_epochs):
            start_ind = 0
            while start_ind < num_samples:
                batch_inds = indices[start_ind: min(num_samples, start_ind + minibatch_size)]
                obs_batch = torch.as_tensor(obs[batch_inds])
                actions_batch = torch.as_tensor(actions[batch_inds])
                log_probs_batch = torch.as_tensor(log_probs[batch_inds])
                reg_ret_batch = torch.as_tensor(reg_ret[batch_inds])
                rnd_ret_batch = torch.as_tensor(rnd_ret[batch_inds])
                adv_batch = torch.as_tensor(adv[batch_inds])
                _, int_loss = rnd.compute_intrinsic(obs_batch)
                metrics['intrinsic_loss'] += to_numpy(int_loss) / (num_samples * num_epochs / minibatch_size)
                rnd_optim.zero_grad()
                int_loss.backward()
                rnd_optim.step()
                
                ppo_loss, metric = compute_rnd_loss(ppo_network, obs_batch, actions_batch, log_probs_batch, reg_ret_batch, rnd_ret_batch, adv_batch,
                                            clip, ent_coef, val_coef)
                for key in metric.keys():
                    if key not in metrics:
                        metrics[key] = 0
                    metrics[key] += to_numpy(metric[key]) / (num_samples * num_epochs / minibatch_size)
                ppo_optim.zero_grad()
                ppo_loss.backward()
                torch.nn.utils.clip_grad_norm_(ppo_network.parameters(), max_grad_norm)
                ppo_optim.step()
                start_ind += minibatch_size
        return metrics
            
            