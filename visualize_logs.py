import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import os

# Path to the newest log file
log_file = "logs/atari_vanilla/events.out.tfevents.1764665905.ip-172-31-39-203.us-east-2.compute.internal.5855.0"

# Load the TensorBoard event file
event_acc = EventAccumulator(log_file)
event_acc.Reload()

# Get available tags
print("Available scalar tags:", event_acc.Tags()['scalars'])

# Extract data for each metric
avg_rew = event_acc.Scalars('AvgRew')
avg_ep_len = event_acc.Scalars('AvgEpLen')
episodes = event_acc.Scalars('Episodes')

# Extract steps and values
steps_rew = [x.step for x in avg_rew]
values_rew = [x.value for x in avg_rew]

steps_len = [x.step for x in avg_ep_len]
values_len = [x.value for x in avg_ep_len]

steps_ep = [x.step for x in episodes]
values_ep = [x.value for x in episodes]

# Create visualization
fig, axes = plt.subplots(3, 1, figsize=(12, 10))

# Plot Average Reward
axes[0].plot(steps_rew, values_rew, linewidth=2, color='#2E86AB')
axes[0].set_xlabel('Training Steps', fontsize=12)
axes[0].set_ylabel('Average Reward', fontsize=12)
axes[0].set_title('Average Reward over Training (Last 100 Episodes)', fontsize=14, fontweight='bold')
axes[0].grid(True, alpha=0.3)

# Plot Average Episode Length
axes[1].plot(steps_len, values_len, linewidth=2, color='#A23B72')
axes[1].set_xlabel('Training Steps', fontsize=12)
axes[1].set_ylabel('Average Episode Length', fontsize=12)
axes[1].set_title('Average Episode Length over Training (Last 100 Episodes)', fontsize=14, fontweight='bold')
axes[1].grid(True, alpha=0.3)

# Plot Total Episodes
axes[2].plot(steps_ep, values_ep, linewidth=2, color='#F18F01')
axes[2].set_xlabel('Training Steps', fontsize=12)
axes[2].set_ylabel('Total Episodes', fontsize=12)
axes[2].set_title('Total Episodes Completed', fontsize=14, fontweight='bold')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_visualization.png', dpi=300, bbox_inches='tight')
print("\nVisualization saved as 'training_visualization.png'")

# Print summary statistics
print(f"\n=== Training Summary ===")
print(f"Total Steps: {steps_rew[-1]:,}")
print(f"Total Episodes: {int(values_ep[-1])}")
print(f"Final Average Reward: {values_rew[-1]:.2f}")
print(f"Max Average Reward: {max(values_rew):.2f}")
print(f"Final Average Episode Length: {values_len[-1]:.2f}")

plt.show()
