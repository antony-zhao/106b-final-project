import numpy as np
import torch

class ReplayBuffer:
    # simple replay buffer for atari testing, probably need to modify for the robot and/or we modify the observations somehow
    def __init__(self, observation_shape, action_dim, obs_type=np.float32, batch_size=64, capacity=1_000_000, device='cuda'):
        self.idx = 0
        self.capacity = capacity
        self.batch_size = batch_size
        self.full = False
        self.device = device
        self.observations = np.empty((capacity, *observation_shape), dtype=obs_type)
        self.actions = np.empty((capacity, *action_dim), dtype=np.float32)
        self.rewards = np.empty((capacity,), dtype=np.float32)
        self.next_observations = np.empty((capacity, *observation_shape), dtype=obs_type)
        self.dones = np.empty((capacity,), dtype=np.bool)
        
    def add_sample(self, obs, action, reward, next_obs, done):
        self.observations[self.idx] = obs
        self.actions[self.idx] = action
        self.rewards[self.idx] = reward
        self.next_observations[self.idx] = next_obs
        self.dones[self.idx] = done
        self.idx += 1
        if self.idx >= self.capacity:
            self.full = True
            self.idx %= self.capacity
    
    def add_batch(self, obs_batch, action_batch, reward_batch, next_obs_batch, done_batch):
        batch_size = obs_batch.shape[0]
        i0 = self.idx
        i1 = self.idx + batch_size
        if i1 <= self.capacity:
            self.observations[i0:i1] = obs_batch
            self.actions[i0:i1] = action_batch
            self.rewards[i0:i1] = reward_batch
            self.next_observations[i0:i1] = next_obs_batch
            self.dones[i0:i1] = done_batch
        else:
            overflow = i1 - self.capacity
            first_part = batch_size - overflow
            self.observations[i0:self.capacity] = obs_batch[:first_part]
            self.actions[i0:self.capacity] = action_batch[:first_part]
            self.rewards[i0:self.capacity] = reward_batch[:first_part]
            self.next_observations[i0:self.capacity] = next_obs_batch[:first_part]
            self.dones[i0:self.capacity] = done_batch[:first_part]
            self.observations[0:overflow] = obs_batch[first_part:]
            self.actions[0:overflow] = action_batch[first_part:]
            self.rewards[0:overflow] = reward_batch[first_part:]
            self.next_observations[0:overflow] = next_obs_batch[first_part:]
            self.dones[0:overflow] = done_batch[first_part:]
        self.idx = (self.idx + batch_size) % self.capacity
        if not self.full and self.idx < i0:
            self.full = True
    
    def sample(self):
        max_idx = self.capacity if self.full else self.idx
        indices = np.random.randint(max_idx, size=self.batch_size)
        
        obs = torch.tensor(self.observations[indices]).to(self.device).float()
        action = torch.tensor(self.actions[indices]).to(self.device).float()
        reward = torch.tensor(self.rewards[indices]).to(self.device).float()
        next_obs = torch.tensor(self.next_observations[indices]).to(self.device).float()
        done = torch.tensor(self.dones[indices]).to(self.device).float()
        
        return obs, action, reward, next_obs, done