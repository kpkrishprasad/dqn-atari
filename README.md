# Deep Q-Network (DQN) and Double DQN for Atari

A PyTorch implementation of Deep Q-Network (DQN) and Double Deep Q-Network (DDQN) algorithms for playing Atari 2600 games. This project implements the architectures and training procedures from:

- **DQN**: "Human-level control through deep reinforcement learning" (Mnih et al., 2015)
- **Double DQN**: "Deep Reinforcement Learning with Double Q-learning" (van Hasselt et al., 2016)

## 🎮 Features

- **Two Algorithm Variants**: Standard DQN and Double DQN implementations
- **Atari Environment Support**: Works with any Atari 2600 game via Gymnasium
- **Efficient Training**: Vectorized environments for faster experience collection
- **Robust Checkpointing**: Corruption-proof checkpoint saving with automatic recovery
- **TensorBoard Integration**: Real-time training visualization
- **Configurable Hyperparameters**: Easy-to-modify config section in training scripts
- **AWS Training Support**: Scripts for training on EC2 instances

## 🏗️ Architecture

### Network Architecture (Nature CNN)
```
Input: 4 stacked 84×84 grayscale frames
↓
Conv2D: 32 filters, 8×8 kernel, stride 4, ReLU
↓
Conv2D: 64 filters, 4×4 kernel, stride 2, ReLU
↓
Conv2D: 64 filters, 3×3 kernel, stride 1, ReLU
↓
Flatten
↓
Fully Connected: 512 units, ReLU
↓
Output Layer: Q-values for each action
```

**Parameters**: ~1.5 million

### Key Differences: DQN vs Double DQN

**DQN Target Computation**:
```
Y_DQN = r + γ · max_a' Q_target(s', a')
```

**Double DQN Target Computation**:
```
Y_DDQN = r + γ · Q_target(s', argmax_a' Q_online(s', a'))
```

Double DQN reduces overestimation bias by decoupling action selection (online network) from action evaluation (target network).

## 📋 Requirements

- Python 3.8+
- PyTorch 2.0+
- Gymnasium with Atari environments
- CUDA-capable GPU (recommended)

## 🚀 Installation

### Local Setup

1. **Clone the repository**:
```bash
git clone https://github.com/kpkrishprasad/dqn-atari.git
cd dqn-atari
```

2. **Create a virtual environment**:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Install Atari ROMs**:
```bash
pip install "gymnasium[atari,accept-rom-license]"
```

### AWS EC2 Setup

For training on AWS EC2 instances:

1. **Transfer files to EC2**:
```bash
# Edit transfer_to_aws.sh with your EC2 details
./transfer_to_aws.sh
```

2. **SSH into EC2 and run setup**:
```bash
ssh -i your-key.pem ec2-user@YOUR_EC2_IP
cd dqn-atari
./setup_aws.sh
```

3. **Activate environment and start training**:
```bash
source venv/bin/activate
python ddqn.py
```

## 🎯 Usage

### Training

The main training script is `ddqn.py`, which supports both DQN and Double DQN:

**Configure your experiment** (edit top of `ddqn.py`):
```python
ENV_ID = "Breakout"        # Game to train on
DOUBLE_DQN = True          # True = DDQN, False = DQN
TOTAL_STEPS = 1_000_000    # Training duration
BUFFER_SIZE = 200_000      # Replay buffer size
MIN_REPLAY_SIZE = 20_000   # Warmup transitions
```

**Start training**:
```bash
python ddqn.py
```

### Key Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Discount factor (γ) | 0.99 | Future reward discount |
| Batch size | 32 | Samples per gradient update |
| Learning rate | 5×10⁻⁵ | Adam optimizer learning rate |
| Replay buffer | 200,000 | Experience replay capacity |
| Warmup size | 20,000 | Random transitions before training |
| Target update freq | 10,000 steps | Target network sync interval |
| ε-greedy start | 1.0 | Initial exploration rate |
| ε-greedy end | 0.1 | Final exploration rate |
| ε decay period | 1,000,000 | Steps to decay epsilon |
| Num environments | 2 | Parallel environment instances |

### Monitoring Training

**Start TensorBoard**:
```bash
tensorboard --logdir=./logsbreaknew
```

Then open http://localhost:6006 in your browser.

**Metrics tracked**:
- Average reward (over last 100 episodes)
- Average episode length
- Episode count
- Training progress

### Resuming Training

Training automatically resumes from the most recent checkpoint if one exists. The checkpoint system is corruption-proof with automatic fallback to backup files.

**Manual checkpoint management**:
```python
CHECKPOINT_INTERVAL = 50_000  # Save every 50k steps
```

### Testing Trained Agents

Watch your trained agent play:
```bash
python observeagain.py
```

Make sure to update the model path in the script to point to your trained model.

## 📊 Experimental Results

### Tested Environments

- **Breakout**: Dense rewards, simple dynamics
- **Fishing Derby**: Sparse rewards, stochastic dynamics

### Performance Comparison

On **Breakout** (1M steps):
- DQN: ~6 average reward
- DDQN: ~6 average reward
- Both algorithms perform similarly

On **Fishing Derby** (1M steps):
- DQN: ~-80 average reward
- DDQN: ~-73 average reward
- DDQN shows ~7 point improvement

**Key Finding**: Double DQN provides significant advantages in environments with stochastic dynamics and sparse rewards (e.g., Fishing Derby) where overestimation bias is more problematic.

## 🗂️ Project Structure

```
dqn-atari/
├── ddqn.py                    # Main training script (DQN + DDQN)
├── dqn.py                     # Original DQN implementation
├── observe.py                 # Watch trained agent play
├── pytorch_wrappers.py        # Custom environment wrappers
├── baselines_wrappers/        # OpenAI Baselines wrappers
├── requirements.txt           # Python dependencies
├── transfer_to_aws.sh         # Transfer code to EC2
├── setup_aws.sh               # Set up environment on EC2
├── download_model_from_aws.sh # Download trained models
├── visualize_logs.py          # Plot training curves
├── compare_fish_derby.py      # Compare DQN vs DDQN results
└── logs/                      # TensorBoard logs
```

## 🔧 Advanced Configuration

### Custom Environments

To train on a different Atari game, simply change:
```python
ENV_ID = "Pong"  # or "SpaceInvaders", "MsPacman", etc.
```

### Modify Network Architecture

Edit the `nature_cnn` function in `ddqn.py`:
```python
def nature_cnn(observation_space, depths=(32, 64, 64), final_layer=512):
    # Modify depths or final_layer size here
    ...
```

### Adjust Training Length

For longer training (as in original paper):
```python
TOTAL_STEPS = 50_000_000  # 50M frames (original paper)
```

Note: You may want to increase `BUFFER_SIZE` to 1,000,000 for longer training runs.

## 📈 Visualization

**Visualize training logs**:
```bash
python visualize_logs.py
```

**Compare DQN vs DDQN**:
```bash
python compare_fish_derby.py
```

This generates comparison plots showing reward progression over training.

## 🐛 Troubleshooting

### CUDA Out of Memory
- Reduce `BATCH_SIZE` (try 16)
- Reduce `NUM_ENVS` (try 1)
- Reduce `BUFFER_SIZE`

### Training is Slow
- Increase `NUM_ENVS` (if memory allows)
- Use a GPU instance for training
- Reduce logging frequency: `LOG_INTERVAL = 10_000`

### Checkpoint Corruption
The checkpoint system automatically handles corruption:
- Saves to temporary file first
- Verifies integrity before overwriting
- Maintains backup of last good checkpoint
- Automatically recovers from `.backup` file if needed

### Poor Performance on Game X
Some games require:
- Longer training (50M steps)
- Larger replay buffer (1M transitions)
- Different exploration schedule
- Reward scaling or clipping adjustments

## 📚 References

1. Mnih, V., Kavukcuoglu, K., Silver, D., et al. (2015). "Human-level control through deep reinforcement learning." *Nature*, 518(7540), 529-533.

2. van Hasselt, H., Guez, A., & Silver, D. (2016). "Deep Reinforcement Learning with Double Q-learning." *AAAI*.

3. Brockman, G., Cheung, V., Pettersson, L., et al. (2016). "OpenAI Gym." *arXiv preprint arXiv:1606.01540*.

## 📄 License

This project is for educational purposes. Please cite the original DQN and Double DQN papers if you use this code in your research.

## 👥 Authors

- Michael Vasandani (mvasandani@ucsd.edu)
- Krish Prasad (krprasad@ucsd.edu)

## 🙏 Acknowledgments

- Implementation inspired by [brthor's DQN tutorial](https://www.youtube.com/watch?v=YOUR_VIDEO_ID)
- OpenAI Baselines for environment wrappers
- DeepMind for the original DQN architecture and training procedures
