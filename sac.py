import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch import vmap

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=256, num_hiddens=3, act=nn.SELU, hidden_dims=None):
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
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding)
        if kernel_size > 1:
            self.skip_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=padding)
        else:
            self.skip_conv = None
        self.act = act()
    
    def forward(self, x):
        x_skip = x.copy()
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        if self.skip_conv is not None:
            x_skip = self.skip_conv(x_skip)
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
            ResBlock(in_channels=128, out_channels=512, kernel_size=3, stride=2) # (128, 7, 7) -> (256, 3, 3) or 2304
        )
        self.output_dim = 2304
    def forward(self, x):
        x = self.convs(x)
        return x

class PolicyNetwork(nn.Module):
    def __init__(self, num_actions):
        super(PolicyNetwork, self).__init__()
        self.conv = AtariConv()
        self.mlp = MLP(self.conv.output_dim, num_actions)
    
    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, self.conv.output_dim)
        x = self.mlp(x)
        return x

class CriticNetwork(nn.Module):
    def __init__(self, num_actions):
        super(CriticNetwork, self).__init__()
        self.conv = AtariConv()
        self.mlp = MLP(self.conv.output_dim + num_actions, num_actions)
    
    def forward(self, x, a):
        x = self.conv(x)
        x = x.view(-1, self.conv.output_dim)
        x = torch.concat([x, a], dim=1)
        x = self.mlp(x)
        return x

class SACNetwork(nn.Module):
    def __init__(self, num_actions, ensemble_size=2, tau=1e-4):
        super(SACNetwork, self).__init__()
        self.tau = tau
        self.ensemble_size = ensemble_size
        self.policy_network = PolicyNetwork(num_actions)
        self.critic_networks = [CriticNetwork(num_actions) for _ in range(ensemble_size)]
        self.target_critics = [CriticNetwork(num_actions) for _ in range(ensemble_size)]
        for critic_network, target_critic in zip(self.critic_networks, self.target_critics):
            target_critic.load_state_dict(critic_network.state_dict())
    
    def compute_Q(self, obs, action):
        x = [critic_network(obs, action) for critic_network in self.critic_networks]
        return torch.stack(x) # as far as i can tell this should result in (num_ensemble, minibatch_size, 1) as the shape
    
    def compute_target_Q(self, obs, action):
        x = [target_critic(obs, action) for target_critic in self.target_critics]
        return torch.stack(x)
    
    def action_dist(self, obs):
        logits = self.policy_network(obs)
        action_dist = Categorical(logits=logits)
        return action_dist
    
    def policy_fn(self, obs):
        # actually choosing the action
        action_dist = self.action_dist(obs)
        action = action_dist.sample()
        return action
    
    def update_target(self):
        for critic_network, target_critic in zip(self.critic_networks, self.target_critics):
            critic_dict = critic_network.state_dict()
            target_dict = target_critic.state_dict()
            for key in critic_dict:
                target_dict[key] = target_dict[key] * (1 - self.tau) + critic_dict[key] * self.tau
            target_critic.load_state_dict(target_dict)

def sac_loss(sac_network: SACNetwork, optim, buffer, discount=0.99, alpha=0.1, critic_coef=1):
    observations, actions, rewards, next_observations, dones = buffer.sample()
    action_dist = sac_network.action_dist(observations)
    next_action_dist = sac_network.action_dist(observations)
    next_action_sample = next_action_dist.sample()
    next_log_prob = action_dist.log_prob(next_action_sample)
    with torch.no_grad():
        next_target_Qs = sac_network.compute_target_Q(next_observations, next_action_sample).min(dim=0)
    Q_targets = rewards + discount * (1 - dones) * (next_target_Qs - alpha * next_log_prob)
    Q_targets = torch.unsqueeze(Q_targets, 0).repeat(2)
    Qs = sac_network.compute_Q(observations, actions)
    critic_loss = F.mse_loss(Qs, Q_targets)
    action_sample = action_dist.sample()
    log_prob = action_dist.log_prob(action_sample)
    Qs_sample = sac_network.compute_Q(observations, action_sample).mean(0)
    policy_loss = -torch.mean(Qs_sample - alpha * log_prob)
    loss = policy_loss + critic_loss * critic_coef
    optim.zero_grad()
    loss.backward()
    optim.step()
    sac_network.update_target()
    metrics = {"critic_loss": critic_loss, "policy_loss": policy_loss}
    return loss, metrics
    
