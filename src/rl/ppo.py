
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
import numpy as np
from distutils.util import strtobool
import time
from torch.utils.tensorboard import summary
import gym

def make_env(gym_id, seed, idx, capture_video, run_name):
    def thunk():
        env = gym.make(gym_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f'video/{run_name}', record_video_trigger=lambda t: t%1000 == 0)
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env
    return thunk

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Agent(nn.Module):
    def __init__(self, envs):
        super(Agent, self).__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).pred(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).pred(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, envs.single_observation_space.n), std=0.01),
        )

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default=os.path.basename(__file__).rstrip('.py'))
    parser.add_argument('--gym_id', type=str, default='Cartpole-v1')
    parser.add_argument('--learning_rate', type=float, default=2.5e-4)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--total_timestamp', type=float, default=25000)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    envs = gym.vector.SyncVectorEnv([make_env(args.gym_id, args.seed+i, i, args.capture_video, run_name)])

    env = gym.make('Cartpole-v1')
    observation = env.reset()
    episodic_return = 0

    for _ in range(200):
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        episodic_return += reward
        if done:
            observation = env.reset()
            episodic_return = 0
    env.close()








