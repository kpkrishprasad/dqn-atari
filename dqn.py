from torch import nn
import os
import torch
import gymnasium as gym
from collections import deque
import itertools
import numpy as np
import random
from pytorch_wrappers import make_atari_deepmind, BatchedPytorchFrameStack, PytorchLazyFrames
from baselines_wrappers import DummyVecEnv, VecEnv, subproc_vec_env, Monitor
from torch.utils.tensorboard import SummaryWriter
import msgpack
from msgpack_numpy import patch as msgpack_numpy_patch
import shutil


msgpack_numpy_patch()

# ============================
# CONFIG / HYPERPARAMETERS
# ============================

# Which Atari game to train on
ENV_ID = "FishingDerby"     # e.g. "Breakout", "FishingDerby", etc.

# Double DQN or vanilla DQN
DOUBLE_DQN = True          # False = normal DQN, True = Double DQN

GAMMA = 0.99
BATCH_SIZE = 32

# Replay buffer
BUFFER_SIZE = 200_000      # can be large; checkpoints will be big
MIN_REPLAY_SIZE = 20_000     # warmup transitions before training

EPSILON_START = 1.0
EPSILON_END = 0.1
EPSILON_DECAY = int(1e6)

NUM_ENVS = 2
TARGET_UPDATE_FREQ = 10000 // NUM_ENVS
LR = 5e-5

# Training length
TOTAL_STEPS = 5_000_000

# Naming based on env + algorithm type
RUN_NAME = f"{ENV_ID}_double{int(DOUBLE_DQN)}"

# Paths
MODELS_DIR = "./modelsfishdouble"
LOG_ROOT = "./logsfishdouble"
CHECKPOINT_DIR = "./checkpointsfishnew"

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOG_ROOT, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

SAVE_PATH = os.path.join(MODELS_DIR, f"{RUN_NAME}.pack")
SAVE_INTERVAL = 50_000  # Save model params every 50k steps

LOG_DIR = os.path.join(LOG_ROOT, f"{RUN_NAME}_lr{LR}")
LOG_INTERVAL = 1_000

CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, f"{RUN_NAME}_checkpoint.pt")
CHECKPOINT_INTERVAL = 50_000  # Save resume checkpoint every 10k steps


# ============================
# CHECKPOINT HELPERS
# ============================

def save_checkpoint(step, online_net, target_net, optimizer,
                    replay_buffer, epinfos_buffer, episode_count):
    """Corruption-proof checkpoint saving."""
    checkpoint = {
        "step": step,
        "online_state_dict": online_net.state_dict(),
        "target_state_dict": target_net.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "replay_buffer": list(replay_buffer),
        "epinfos_buffer": list(epinfos_buffer),
        "episode_count": episode_count,
        "rng_state": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state().cpu(),  # <-- Added .cpu()
            "torch_cuda": [s.cpu() for s in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else None,  # <-- Added .cpu()
        },
    }
    
    tmp_path = CHECKPOINT_PATH + ".tmp"
    backup_path = CHECKPOINT_PATH + ".backup"
    
    # 1. Save to temp file
    torch.save(checkpoint, tmp_path)
    
    # 2. Verify temp file is valid
    try:
        torch.load(tmp_path, map_location="cpu")
    except Exception as e:
        print(f"[Checkpoint] Verification failed, keeping old checkpoint: {e}")
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        return
    
    # 3. Backup current good checkpoint (if exists)
    if os.path.exists(CHECKPOINT_PATH):
        shutil.copy2(CHECKPOINT_PATH, backup_path)
    
    # 4. Atomic rename temp -> final
    os.replace(tmp_path, CHECKPOINT_PATH)
    print(f"[Checkpoint] Saved at step {step} -> {CHECKPOINT_PATH}")  


def load_checkpoint(device):
    """Load checkpoint with fallback to backup."""
    paths_to_try = [CHECKPOINT_PATH, CHECKPOINT_PATH + ".backup"]
    
    for path in paths_to_try:
        if not os.path.exists(path):
            continue
        
        try:
            checkpoint = torch.load(path, map_location=device)
            print(f"[Checkpoint] Loaded from {path} at step {checkpoint['step']}")
            
            # Restore RNG states
            rng = checkpoint.get("rng_state", None)
            if rng is not None:
                random.setstate(rng["python"])
                np.random.set_state(rng["numpy"])
                
                # RNG state MUST be ByteTensor on CPU
                torch_rng = rng["torch"]
                if isinstance(torch_rng, torch.Tensor):
                    torch_rng = torch_rng.cpu().byte()
                else:
                    torch_rng = torch.ByteTensor(torch_rng)
                torch.set_rng_state(torch_rng)
                
                if torch.cuda.is_available() and rng["torch_cuda"] is not None:
                    cuda_rng = []
                    for s in rng["torch_cuda"]:
                        if isinstance(s, torch.Tensor):
                            cuda_rng.append(s.cpu().byte())
                        else:
                            cuda_rng.append(torch.ByteTensor(s))
                    torch.cuda.set_rng_state_all(cuda_rng)
            
            return checkpoint
        
        except Exception as e:
            print(f"[Checkpoint] Failed to load {path}: {e}")
            continue
    
    print("[Checkpoint] No valid checkpoint found, starting from scratch.")
    return None


# ============================
# NETWORK ARCHITECTURE
# ============================

def nature_cnn(observation_space, depths=(32, 64, 64), final_layer=512):
    # Observation space is channel-first
    n_input_channels = observation_space.shape[0]

    cnn = nn.Sequential(
        nn.Conv2d(n_input_channels, depths[0], kernel_size=8, stride=4),
        nn.ReLU(),
        nn.Conv2d(depths[0], depths[1], kernel_size=4, stride=2),
        nn.ReLU(),
        nn.Conv2d(depths[1], depths[2], kernel_size=3, stride=1),
        nn.ReLU(),
        nn.Flatten()
    )

    # Compute flattened size with a dummy forward pass
    with torch.no_grad():
        n_flatten = cnn(
            torch.as_tensor(observation_space.sample()[None]).float()).shape[1]

    out = nn.Sequential(
        cnn,
        nn.Linear(n_flatten, final_layer),
        nn.ReLU()
    )

    return out


class Network(nn.Module):
    def __init__(self, env, device, double=True):
        super().__init__()

        self.num_actions = env.action_space.n
        self.device = device
        self.double = double

        conv_net = nature_cnn(env.observation_space)

        self.net = nn.Sequential(
            conv_net,
            nn.Linear(512, self.num_actions)
        )

    def forward(self, x):
        return self.net(x)

    def act(self, obses, epsilon):
        """Epsilon-greedy action selection for a batch of observations."""
        if isinstance(obses, list):
            obses = np.array(obses, dtype=np.float32)
        obses_t = torch.as_tensor(obses, dtype=torch.float32, device=self.device)
        q_values = self(obses_t)
        max_q_indices = torch.argmax(q_values, dim=1)
        actions = max_q_indices.detach().tolist()

        for i in range(len(actions)):
            rnd_sample = random.random()
            if rnd_sample <= epsilon:
                actions[i] = random.randint(0, self.num_actions - 1)

        return actions

    def compute_loss(self, transitions, target_net):
        # Unpack batch
        obses = [t[0] for t in transitions]
        actions = np.asarray([t[1] for t in transitions])
        rews = np.asarray([t[2] for t in transitions])
        dones = np.asarray([t[3] for t in transitions])
        new_obses = [t[4] for t in transitions]

        if isinstance(obses[0], PytorchLazyFrames):
            obses = np.stack([o.get_frames() for o in obses])
            new_obses = np.stack([o.get_frames() for o in new_obses])
        else:
            obses = np.asarray(obses)
            new_obses = np.asarray(new_obses)

        # To tensors
        obses_t = torch.as_tensor(obses, dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(-1)
        rews_t = torch.as_tensor(rews, dtype=torch.float32, device=self.device).unsqueeze(-1)
        dones_t = torch.as_tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(-1)
        new_obses_t = torch.as_tensor(new_obses, dtype=torch.float32, device=self.device)

        # Compute Targets
        with torch.no_grad():
            if self.double:
                # Double DQN: use online net to select action, target net to evaluate
                target_online_q_values = self(new_obses_t)
                targets_online_best_q_indices = target_online_q_values.argmax(dim=1, keepdim=True)
                targets_target_q_values = target_net(new_obses_t)
                targets_selected_q_values = torch.gather(
                    input=targets_target_q_values,
                    dim=1,
                    index=targets_online_best_q_indices
                )
                targets = rews_t + GAMMA * (1 - dones_t) * targets_selected_q_values
            else:
                # Vanilla DQN: directly take max over target net
                target_q_values = target_net(new_obses_t)
                max_target_q_values = target_q_values.max(dim=1, keepdim=True)[0]
                targets = rews_t + GAMMA * (1 - dones_t) * max_target_q_values

        # Compute Loss
        q_values = self(obses_t)
        action_q_values = torch.gather(input=q_values, dim=1, index=actions_t)
        loss = nn.functional.smooth_l1_loss(action_q_values, targets)
        return loss

    def save(self, save_path):
        """Your original msgpack-based save (for homework)."""
        params = {k: t.detach().cpu().numpy() for k, t in self.state_dict().items()}
        params_data = msgpack.packb(params)

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as f:
            f.write(params_data)

    def load(self, load_path):
        if not os.path.exists(load_path):
            raise FileNotFoundError(load_path)

        with open(load_path, 'rb') as f:
            params_numpy = msgpack.unpackb(f.read())

        params = {k: torch.as_tensor(v, device=self.device) for k, v in params_numpy.items()}
        self.load_state_dict(params)


# ============================
# MAIN TRAINING SCRIPT
# ============================

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    print(f"Environment: {ENV_ID}, Double DQN: {DOUBLE_DQN}")
    print(f"Replay buffer size: {BUFFER_SIZE}, Warmup: {MIN_REPLAY_SIZE}")
    print(f"Total steps: {TOTAL_STEPS}")

    # Env setup
    make_env = lambda: Monitor(
        make_atari_deepmind(ENV_ID, scale_values=True),
        filename=None,
        allow_early_resets=True
    )
    vec_env = DummyVecEnv([make_env for _ in range(NUM_ENVS)])
    env = BatchedPytorchFrameStack(vec_env, k=4)

    # Buffers & logging
    replay_buffer = deque(maxlen=BUFFER_SIZE)
    epinfos_buffer = deque([], maxlen=100)
    episode_count = 0.0

    summary_writer = SummaryWriter(LOG_DIR)

    # Networks + optimizer
    online_net = Network(env, device=device, double=DOUBLE_DQN).to(device)
    target_net = Network(env, device=device, double=DOUBLE_DQN).to(device)
    optimizer = torch.optim.Adam(online_net.parameters(), lr=LR)

    # Try to resume from checkpoint
    checkpoint = load_checkpoint(device)
    if checkpoint is not None:
        online_net.load_state_dict(checkpoint["online_state_dict"])
        target_net.load_state_dict(checkpoint["target_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        replay_buffer = deque(checkpoint.get("replay_buffer", []), maxlen=BUFFER_SIZE)
        epinfos_buffer = deque(checkpoint.get("epinfos_buffer", []), maxlen=100)
        episode_count = checkpoint.get("episode_count", 0.0)

        start_step = checkpoint["step"] + 1
        print(f"Resuming training from step {start_step}, replay size {len(replay_buffer)}")
    else:
        target_net.load_state_dict(online_net.state_dict())
        start_step = 0
        print("Starting training from scratch")

        # Warmup replay buffer only when no checkpoint
        print(f"Filling replay buffer with {MIN_REPLAY_SIZE} transitions...")
        obses = env.reset()
        while len(replay_buffer) < MIN_REPLAY_SIZE:
            actions = [env.action_space.sample() for _ in range(NUM_ENVS)]
            new_obses, rews, dones, _ = env.step(actions)
            for obs, action, rew, done, new_obs in zip(obses, actions, rews, dones, new_obses):
                replay_buffer.append((obs, action, rew, done, new_obs))
            obses = new_obses
        print("Warmup complete.")

    # Main training loop
    obses = env.reset()

    for step in range(start_step, TOTAL_STEPS):
        epsilon = np.interp(step, [0, EPSILON_DECAY], [EPSILON_START, EPSILON_END])

        if isinstance(obses[0], PytorchLazyFrames):
            act_obses = np.stack([o.get_frames() for o in obses])
            actions = online_net.act(act_obses, epsilon)
        else:
            actions = online_net.act(obses, epsilon)

        new_obses, rews, dones, infos = env.step(actions)

        for obs, action, rew, done, new_obs, info in zip(
            obses, actions, rews, dones, new_obses, infos
        ):
            replay_buffer.append((obs, action, rew, done, new_obs))
            if done:
                if "episode" in info:
                    epinfos_buffer.append(info["episode"])
                episode_count += 1

        obses = new_obses

        # Gradient step
        transitions = random.sample(replay_buffer, BATCH_SIZE)
        loss = online_net.compute_loss(transitions, target_net)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Target network update
        if step % TARGET_UPDATE_FREQ == 0:
            target_net.load_state_dict(online_net.state_dict())

        # Logging (only if we have enough episodes)
        if step % LOG_INTERVAL == 0 and len(epinfos_buffer) >= 10:
            rew_mean = float(np.mean([e['r'] for e in epinfos_buffer]))
            len_mean = float(np.mean([e['l'] for e in epinfos_buffer]))

            print()
            print('Step', step)
            print('Avg Reward', rew_mean)
            print('Avg Episode Len', len_mean)
            print('Episodes (total)', episode_count)
            print('Episodes in buffer', len(epinfos_buffer))

            summary_writer.add_scalar('AvgRew', rew_mean, global_step=step)
            summary_writer.add_scalar('AvgEpLen', len_mean, global_step=step)
            summary_writer.add_scalar('Episodes', episode_count, global_step=step)

        # Checkpoint (for resume)
        # if step % CHECKPOINT_INTERVAL == 0 and step != 0:
        #     save_checkpoint(
        #         step, online_net, target_net, optimizer,
        #         replay_buffer, epinfos_buffer, episode_count
        #     )

        # Homework-style param save
        if step % SAVE_INTERVAL == 0 and step != 0:
            print('Saving model params ...')
            online_net.save(SAVE_PATH)

    print("Training finished.")