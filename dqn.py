from torch import nn
import torch
import gymnasium as gym
from collections import deque
import itertools
import numpy as np
import random
from pytorch_wrappers import make_atari_deepmind
from baselines_wrappers import DummyVecEnv, VecEnv


GAMMA = 0.99
BATCH_SIZE = 32
BUFFER_SIZE = 50000
MIN_REPLAY_SIZE = 1000
EPSILON_START = 1.0
EPSILON_END = 0.02
EPSILON_DECAY = 10000
TARGET_UPDATE_FREQ = 1000
NUM_ENVS = 4

class Network(nn.Module):
    def __init__(self, env):
        super().__init__()
        in_features = int(np.prod(env.observation_space.shape))

        self.net = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.Tanh(),
            nn.Linear(64, env.action_space.n)            
        )

    def forward(self, x):
        return self.net(x)
    
    def act(self, obs):
        obs_t = torch.as_tensor(obs, dtype = torch.float32)
        q_values = self(obs_t.unsqueeze(0))
        max_q_index = torch.argmax(q_values, dim = 1)[0]
        action = max_q_index.detach().item()

        return action
    

make_env = lambda: make_atari_deepmind('Breakout-v0')
env = DummyVecEnv([make_env for _ in range(NUM_ENVS)])



replay_buffer = deque(maxlen = BUFFER_SIZE)
rew_buffer = deque([0, 0], maxlen = 100)

episode_reward = 0.0

online_net = Network(env)
target_net = Network(env)

target_net.load_state_dict(online_net.state_dict())

optimizer = torch.optim.Adam(online_net.parameters(), lr = 5e-4)

#Replay Buffer
obs, _ = env.reset()
for _ in range(MIN_REPLAY_SIZE):
    action = env.action_space.sample()
    new_obs, rew, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    transition = (obs, action, rew, done, new_obs)
    replay_buffer.append(transition)

    if done:
        obs, _ = env.reset()

#Main Training Loop
obs, _ = env.reset()

for step in itertools.count():
    epsilon = np.interp(step, [0, EPSILON_DECAY], [EPSILON_START, EPSILON_END])
    rnd_sample = random.random()

    if rnd_sample <= epsilon:
        action = env.action_space.sample()
    else:
        action = online_net.act(obs)

    new_obs, rew, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
    transition = (obs, action, rew, done, new_obs)
    replay_buffer.append(transition)
    obs = new_obs

    episode_reward += rew

    if done:
        obs, _ = env.reset()
        rew_buffer.append(episode_reward)
        episode_reward = 0.0

    #After solved, watch it play

    if len(rew_buffer) >= 100:
        if np.mean(rew_buffer) >= 195:
            env_render = gym.make('CartPole-v1', render_mode='human')
            obs_render, _ = env_render.reset()
            for _ in range(500):
                action = online_net.act(obs_render)
                obs_render, _, terminated, truncated, _ = env_render.step(action)
                done = terminated or truncated
                if done:
                    obs_render, _ = env_render.reset()
            break


    transitions = random.sample(replay_buffer, BATCH_SIZE)
    obses      = np.asarray([t[0] for t in transitions])
    actions    = np.asarray([t[1] for t in transitions])
    rewards    = np.asarray([t[2] for t in transitions])
    dones      = np.asarray([t[3] for t in transitions])
    new_obses = np.asarray([t[4] for t in transitions])

    obses_t      = torch.as_tensor(obses, dtype=torch.float32)
    actions_t    = torch.as_tensor(actions, dtype=torch.int64).unsqueeze(-1)
    rews_t       = torch.as_tensor(rewards, dtype=torch.float32).unsqueeze(-1)
    dones_t      = torch.as_tensor(dones, dtype=torch.float32).unsqueeze(-1)
    new_obses_t  = torch.as_tensor(new_obses, dtype=torch.float32)

        # Compute Targets
    target_q_values = target_net(new_obses_t)
    max_target_q_values = target_q_values.max(dim=1, keepdim=True)[0]

    targets = rews_t + GAMMA * (1 - dones_t) * max_target_q_values

    # Compute Loss
    q_values = online_net(obses_t)

    action_q_values = torch.gather(
        input=q_values,
        dim=1,
        index=actions_t
    )

    loss = nn.functional.smooth_l1_loss(action_q_values, targets)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % TARGET_UPDATE_FREQ == 0:
        target_net.load_state_dict(online_net.state_dict())

    if step % 1000 == 0:
        print()
        print('Step', step)
        print('Avg Rew', np.mean(rew_buffer))



