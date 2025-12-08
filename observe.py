import numpy as np
import torch
import itertools
from baselines_wrappers import DummyVecEnv
from dqn import Network, SAVE_PATH
from pytorch_wrappers import make_atari_deepmind, BatchedPytorchFrameStack, PytorchLazyFrames
import time

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('device:', device)

import gymnasium as gym
from baselines_wrappers.atari_wrappers import NoopResetEnv, MaxAndSkipEnv, EpisodicLifeEnv, WarpFrame, ClipRewardEnv, ScaledFloatFrame
from pytorch_wrappers import TransposeImageObs

def make_env_render():
    env = gym.make('ALE/Breakout-v5', render_mode='human')
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    env = EpisodicLifeEnv(env)
    env = WarpFrame(env)
    env = ScaledFloatFrame(env)  # IMPORTANT: Scale values to match training!
    env = ClipRewardEnv(env)
    env = TransposeImageObs(env, op=[2, 0, 1])  # Convert to torch order (C, H, W)
    return env

make_env = make_env_render

vec_env = DummyVecEnv([make_env for _ in range(1)])

env = BatchedPytorchFrameStack(vec_env, k=4)

net = Network(env, device)
net = net.to(device)

net.load(SAVE_PATH)

obs = env.reset()
beginning_episode = True
for t in itertools.count():
    if isinstance(obs[0], PytorchLazyFrames):
        act_obs = np.stack([o.get_frames() for o in obs])
        action = net.act(act_obs, 0.0)
    else:
        action = net.act(obs, 0.0)
    
    if beginning_episode:
        action = [1]
        beginning_episode = False
    
    obs, rew, done, _ = env.step(action)
    time.sleep(0.02)  # Small delay to control frame rate
    
    if done[0]:
        obs = env.reset()
        beginning_episode = True
