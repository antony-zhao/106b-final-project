import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Categorical
import copy

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=256, num_hiddens=2, act=nn.SELU, hidden_dims=None):
        super(MLP, self).__init__()
        if hidden_dims is not None:
            assert len(hidden_dims) + 1 == num_hiddens
            hidden_dim = hidden_dims[0]
            self.skip_connections = False
        else:
            self.skip_connections = True
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.hiddens = []
        for i in range(num_hiddens):
            if hidden_dims is None:
                self.hiddens.append(nn.Linear(hidden_dim, hidden_dim))
            else:
                self.hiddens.append(nn.Linear(hidden_dim, hidden_dims[i + 1]))
                hidden_dim = hidden_dims[i + 1]
        self.hiddens = nn.ModuleList(self.hiddens)
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.act = act()
    
    def forward(self, x):
        x = self.act(self.input_layer(x))
        for i in range(len(self.hiddens)):
            if self.skip_connections:
                x = self.act(self.hiddens[i](x)) + x
            else:
                x = self.act(self.hiddens[i](x))
        logits = self.output_layer(x)
        return logits

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, act=nn.SELU):
        super(ResBlock, self).__init__()
        padding = int((kernel_size - 1) // 2)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, 1, padding)
        if stride > 1:
            self.skip = nn.Conv2d(in_channels, out_channels, 1, stride)
        else:
            self.skip = None
        self.act = act()
    
    def forward(self, x):
        x_skip = x.clone()
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        if self.skip is not None:
            x_skip = self.skip(x_skip)
        return x + x_skip
        
class AtariConv(nn.Module):
    # assumes the 84x84 grayscale and 4 frame stack
    def __init__(self, act=nn.SELU):
        super(AtariConv, self).__init__()
        self.convs = nn.Sequential(
            ResBlock(in_channels=4, out_channels=32, kernel_size=7, stride=3), # (4, 84, 84) -> (32, 28, 28)
            act(),
            ResBlock(in_channels=32, out_channels=64, kernel_size=5, stride=2), # (32, 28, 28) -> (64, 14, 14)
            act(),
            ResBlock(in_channels=64, out_channels=128, kernel_size=3, stride=2), # (64, 14, 14) -> (128, 7, 7)
            act(),
            ResBlock(in_channels=128, out_channels=256, kernel_size=3, stride=2) # (128, 7, 7) -> (256, 4, 4) or 4096
        )
        self.output_dim = 4096
    def forward(self, x):
        x = self.convs(x)
        return x

class AtariActorNetwork(nn.Module):
    def __init__(self, num_actions):
        super(AtariActorNetwork, self).__init__()
        self.conv = AtariConv()
        self.mlp = MLP(self.conv.output_dim, num_actions)
    
    def forward(self, x):
        x = self.conv(x)
        x = x.reshape(-1, self.conv.output_dim)
        x = self.mlp(x)
        return x
    
    def action_dist(self, x):
        logits = self(x)
        action_dist = Categorical(logits=logits)
        return action_dist
    
    def act(self, x):
        # actually choosing the action
        action_dist = self.action_dist(x)
        action = action_dist.sample()
        return action

class AtariCriticNetwork(nn.Module):
    def __init__(self, num_actions):
        super(AtariCriticNetwork, self).__init__()
        self.conv = AtariConv()
        self.mlp = MLP(self.conv.output_dim, num_actions)
    
    def forward(self, x, a):
        x = self.conv(x)
        x = x.view(-1, self.conv.output_dim)
        x = self.mlp(x)
        return x

class SACNetwork(nn.Module):
    def __init__(self, actor_network, critic_networks, tau=1e-4):
        super(SACNetwork, self).__init__()
        self.tau = tau
        self.ensemble_size = len(critic_networks)
        self.actor_network = actor_network
        self.critic_networks = critic_networks
        self.target_critics = copy.deepcopy(critic_networks)
        for critic_network, target_critic in zip(self.critic_networks, self.target_critics):
            target_critic.load_state_dict(critic_network.state_dict())
    
    def compute_Q(self, obs, action=None):
        x = [critic_network(obs, action) for critic_network in self.critic_networks]
        return torch.stack(x) # as far as i can tell this should result in (num_ensemble, minibatch_size, 1) as the shape
    
    def compute_target_Q(self, obs, action=None):
        x = [target_critic(obs, action) for target_critic in self.target_critics]
        return torch.stack(x)
    
    def action_dist(self, obs):
        return self.actor_network.action_dist(obs)
    
    def act(self, obs):
        return self.actor_network.act(obs)
    
    def update_target(self):
        for critic_network, target_critic in zip(self.critic_networks, self.target_critics):
            critic_dict = critic_network.state_dict()
            target_dict = target_critic.state_dict()
            for key in critic_dict:
                target_dict[key] = target_dict[key] * (1 - self.tau) + critic_dict[key] * self.tau
            target_critic.load_state_dict(target_dict)

def sac_loss(sac_network: SACNetwork, actor_optim, critic_optim, buffer, discount=0.99, alpha=0.1, critic_coef=1, max_grad_norm=0.5):
    observations, actions, rewards, next_observations, dones = buffer.sample()
    
    action_dist = sac_network.action_dist(observations)
    next_action_dist = sac_network.action_dist(next_observations)
    next_action_sample = next_action_dist.sample()
    next_log_prob = next_action_dist.log_prob(next_action_sample)
    with torch.no_grad():
        next_target_Qs = sac_network.compute_target_Q(next_observations, next_action_sample).min(dim=0)[0].squeeze(1)
    Q_targets = rewards + discount * (1 - dones) * (next_target_Qs - alpha * next_log_prob)
    Q_targets = torch.unsqueeze(Q_targets, 0).repeat(sac_network.ensemble_size, 1)
    Qs = sac_network.compute_Q(observations, actions).squeeze(-1)
    critic_loss = F.mse_loss(Qs, Q_targets)
    critic_optim.zero_grad()
    critic_loss.backward()
    nn.utils.clip_grad_norm_(sac_network.critic_networks.parameters(), max_norm=max_grad_norm)
    critic_optim.step()
    
    action_sample = action_dist.sample()
    log_prob = action_dist.log_prob(action_sample)
    Qs_sample = sac_network.compute_Q(observations, action_sample).mean(0).squeeze(-1)
    actor_loss = -torch.mean(Qs_sample - alpha * log_prob)
    loss = actor_loss + critic_loss * critic_coef
    actor_optim.zero_grad()
    actor_loss.backward()
    nn.utils.clip_grad_norm_(sac_network.actor_network.parameters(), max_norm=max_grad_norm)
    actor_optim.step()
    sac_network.update_target()
    
    metrics = {"critic_loss": critic_loss.cpu().detach().numpy(), "actor_loss": actor_loss.cpu().detach().numpy()}
    return loss.cpu().detach().numpy(), metrics
    
def discrete_sac_loss(sac_network: SACNetwork, actor_optim, critic_optim, buffer, discount=0.99, alpha=0.1, critic_coef=1, max_grad_norm=0.5):
    observations, actions, rewards, next_observations, dones = buffer.sample()
    
    action_dist = sac_network.action_dist(observations)
    next_action_dist = sac_network.action_dist(next_observations)
    next_prob = next_action_dist.probs
    with torch.no_grad():
        next_target_Qs = sac_network.compute_target_Q(next_observations).min(dim=0)[0]
    Q_targets = rewards + discount * (1 - dones) * torch.sum(next_prob * (next_target_Qs - alpha * torch.log(next_prob + 1e-8)), dim=1)
    Q_targets = torch.unsqueeze(Q_targets, 0).repeat(sac_network.ensemble_size, 1)
    Qs = sac_network.compute_Q(observations).gather(2, actions.long().unsqueeze(0).unsqueeze(-1).repeat(sac_network.ensemble_size, 1, 1)).squeeze(-1)
    critic_loss = F.mse_loss(Qs, Q_targets)
    critic_optim.zero_grad()
    critic_loss.backward()
    nn.utils.clip_grad_norm_(sac_network.critic_networks.parameters(), max_norm=max_grad_norm)
    critic_optim.step()
    
    prob = action_dist.probs
    Qs_sample = sac_network.compute_Q(observations).mean(0)
    actor_loss = -torch.mean(torch.sum(prob * (Qs_sample - alpha * torch.log(prob + 1e-8)), dim=1))
    loss = actor_loss + critic_loss * critic_coef
    actor_optim.zero_grad()
    actor_loss.backward()
    nn.utils.clip_grad_norm_(sac_network.actor_network.parameters(), max_norm=max_grad_norm)
    actor_optim.step()
    sac_network.update_target()
    
    metrics = {"critic_loss": critic_loss.cpu().detach().numpy(), "actor_loss": actor_loss.cpu().detach().numpy()}
    return loss.cpu().detach().numpy(), metrics
    
