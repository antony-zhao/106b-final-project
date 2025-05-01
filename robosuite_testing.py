import numpy as np
import robosuite as suite
import gymnasium as gym
from robosuite.wrappers import GymWrapper, DomainRandomizationWrapper
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from rnd import RND, RunningMeanStd, PPONetwork, train_rnd, RND_Rollout, layer_init
from utils import to_numpy
from logger import Logger
import imageio
from models import ResBlock, MLP

class RobosuiteCore(nn.Module):
    def __init__(self, hidden=64, camera_dim=64, framestack=4, proprio_dim=43, act=nn.SELU, num_hidden=3):
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

def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)    
    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    
    # create environment instance
    def make_env(env_name, camera_dim, seed, framestack=4, eval=False):
        def thunk():
            env = suite.make(
                env_name=env_name, # try with other tasks like "Stack" and "Door"
                robots="Sawyer",  # try with other robots like "Sawyer" and "Jaco"
                # has_renderer=eval,
                render_collision_mesh=False,
                has_offscreen_renderer=True,
                use_camera_obs=True,
                camera_names=["agentview", 'frontview'],
                use_object_obs=False,
                # object_type='can',
                # single_object_mode=2,
                camera_heights=camera_dim,
                camera_widths=camera_dim,
                reward_shaping=True,
                hard_reset=False,
                horizon=256,
                control_freq=10,
                table_full_size=(0.8, 2.0, 0.05)
            )
            # if not eval:
            env = DomainRandomizationWrapper(env, seed=seed, randomize_color=False, randomize_every_n_steps=0)
            env = GymWrapper(env, flatten_obs=False)
            # env.render_mode = 'rgb_array'
            env = gym.wrappers.TransformObservation(env, 
                                                    transform_obs, 
                                                    gym.spaces.Box(-np.inf, np.inf, shape=(camera_dim, camera_dim, 2))
                                                    )
            env = gym.wrappers.FrameStackObservation(env, framestack)
            env = gym.wrappers.TransformObservation(env, 
                                                    transform_framestacked_obs, 
                                                    gym.spaces.Box(-np.inf, np.inf, shape=(2 * framestack, camera_dim, camera_dim))
                                                    )
            env = gym.wrappers.RecordEpisodeStatistics(env)
            env.action_space.seed(seed)
            env.observation_space.seed(seed)
            return env
        return thunk


    # reset the environment
    # plt.imsave('test.png', np.flip(obs['agentview_image'], 0))
    '''
    keys:
    odict_keys(['robot0_joint_pos', 'robot0_joint_pos_cos', 'robot0_joint_pos_sin', 'robot0_joint_vel', 
    'robot0_eef_pos', 'robot0_eef_quat', 'robot0_eef_quat_site', 
    'robot0_gripper_qpos', 'robot0_gripper_qvel', 43 inputs
    'agentview_image', 
    'robot0_proprio-state', 'object-state'])
    robot0_proprio-state uses 'robot0_joint_pos', 'robot0_joint_pos_cos', 'robot0_joint_pos_sin', 'robot0_joint_vel', 
    'robot0_eef_pos', 'robot0_eef_quat', 'robot0_eef_quat_site', 
    'robot0_gripper_qpos', 'robot0_gripper_qvel' in order
    agentview_image is 100, 100, 3
    '''

    def transform_obs(obs, camera_name='agentview_image'):
        image = np.flip(obs[camera_name], 0).mean(-1, keepdims=True) / 255
        proprio = obs['robot0_proprio-state'] * 10
        dim = image.shape[0] * image.shape[1]
        new_channel = np.zeros(dim)
        new_channel[:proprio.size] = proprio
        new_channel = new_channel.reshape(image.shape[0], image.shape[1], 1)
        new_obs = np.concatenate([image, new_channel], axis=-1)
        return new_obs

    def transform_framestacked_obs(obs):
        _, h, w, _ = obs.shape
        obs = obs.transpose(0, 3, 1, 2)
        obs = obs.reshape(-1, h, w)
        return obs
    
    env = make_vec_env(make_env(args.env, args.camera_dim, args.seed, args.framestack), args.num_envs, args.seed, vec_env_cls=SubprocVecEnv, vec_env_kwargs=dict(start_method='spawn'))
    eval_env = make_env(args.env, args.camera_dim, args.seed, args.framestack, eval=True)()
    
    def eval(ppo_network, eval_env):
        done = False
        obs, _ = eval_env.reset()
        total_reward = 0
        num_completed = 0
        frames = []
        while not done:
            obs = torch.as_tensor(obs).to(device).float()
            action, _ = ppo_network.policy_network.policy_fn(obs, det=True)
            # action[:, 0] = 0
            action = torch.tanh(action)
            obs, reward, term, trunc, infos = eval_env.step(to_numpy(action))
            done = term or trunc
            total_reward += reward
            frames.append(np.flip(eval_env.env.env.env.env._get_observations()['agentview_image'], 0))
            for info in infos:
                if 'episode' in info:
                    total_reward += infos['episode']['r']
                    num_completed += 1
        env.reset()
        return total_reward, frames

    def anneal_lr(optim, lr, update, num_updates):
        frac = 1.0 - (update) / num_updates
        new_lr = frac * lr
        optim.param_groups[0]["lr"] = new_lr    
        
    with torch.device(device):
        feature_model = RobosuiteFeatureModel(num_hidden=3, camera_dim=args.camera_dim)
        target_model = RobosuiteFeatureModel(num_hidden=0, camera_dim=args.camera_dim)
        rnd_opt = optim.Adam(feature_model.parameters(), lr=args.lr)
        rnd = RND(feature_model, target_model, obs_channels=2)
        
        policy = RobosuitePolicy(env.action_space.shape[0], camera_dim=args.camera_dim, framestack=args.framestack)
        value = RobosuiteValue(camera_dim=args.camera_dim, framestack=args.framestack)
        layers = [module for module in policy.modules() if isinstance(module, (nn.Linear, nn.Conv2d))]
        for i, module in enumerate(layers):
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                if i < len(layers) - 1:
                  layer_init(module)
                else:
                    layer_init(module, std=0.01)
        layers = [module for module in value.modules() if isinstance(module, (nn.Linear, nn.Conv2d))]
        for i, module in enumerate(layers):
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                # if i < len(layers) - 1:
                layer_init(module)
                # else:
                #     layer_init(module, std=1.0)
        ppo_network = PPONetwork(policy, value)
        for module in rnd.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                layer_init(module)
        ppo_opt = optim.Adam(ppo_network.parameters(), lr=args.lr, eps=1e-5)
    
        obs_rms = RunningMeanStd()
        rnd_rms = RunningMeanStd()
        obs = env.reset()
        for _ in range(50):
            action = [env.action_space.sample() for _ in range(args.num_envs)]
            obs, _, _, _ = env.step(action)
            obs_rms.update(obs)
        
        total_updates = args.timesteps // (args.num_envs * args.rollout_length)
        logger = Logger(f'logs/{args.env}')
        for i in range(total_updates):
            rollout, obs, train_rew = collect_rollout(env, ppo_network, rnd, args.rollout_length, obs, obs_rms)
            metrics = train_rnd(rollout, ppo_network, rnd, obs_rms, rnd_rms, ppo_opt, rnd_opt, args.num_minibatches, args.num_epochs, device, rnd_coef=0, discount=0.99,
                                max_grad_norm=args.max_grad_norm, clip=args.clip, ent_coef=args.ent_coef, val_coef=args.val_coef, sample_prob=32 / args.num_envs)
            anneal_lr(ppo_opt, args.lr, i, total_updates)
            logger.add_metrics(metrics)
            print(f'{i * args.num_envs * args.rollout_length}: {train_rew}')
            if train_rew is not None:
                logger.add_scalar('train_reward', train_rew)
            if i % 10 == 0:
                eval_reward, frames = eval(ppo_network, eval_env)
                print(f'epoch {i}: {np.mean(eval_reward)}')
                logger.add_scalar('eval_reward', eval_reward)
                imageio.mimwrite(f'gifs/{args.env}_{i}.gif', frames[::4], loop=0, fps=20)
                torch.save(ppo_network.state_dict(), f'{logger.logdir}/ppo_{i}.pt')
                # np.savez(f'{logger.logdir}/proprio_norm_{i}.npz', mean=proprio_rms.mean, var=proprio_rms.var, count=proprio_rms.count)
            logger.write(i * args.num_envs * args.rollout_length)

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--env', type=str, default='Lift') 
    parser.add_argument('--camera-dim', type=int, default=100)
    parser.add_argument('--discount', type=float, default=0.99)
    parser.add_argument('--clip', type=float, default=0.1)
    parser.add_argument('--val-coef', type=float, default=0.5)
    parser.add_argument('--ent-coef', type=float, default=1e-3)
    parser.add_argument('--max-grad-norm', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--rollout-length', type=int, default=256)
    parser.add_argument('--num-envs', type=int, default=32)
    parser.add_argument('--timesteps', type=int, default=500_000_000)
    parser.add_argument('--num-epochs', type=int, default=4)
    parser.add_argument('--num-minibatches', type=int, default=4)
    parser.add_argument('--framestack', type=int, default=4)
    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_args()
    main(args)