import matplotlib.pyplot as plt
import numpy as np
import struct
import tensorflow as tf
from collections import defaultdict

# Paths to the event files
dqn_file = "logs_visualize/fish_derby_dqn.0"
ddqn_file = "logs_visualize/fish_derby_ddqn.0"

def parse_tfevents_file(filepath):
    """Parse TensorFlow events file and extract scalar summaries."""
    data = defaultdict(lambda: {'steps': [], 'values': []})

    print(f"Parsing {filepath}...")

    try:
        # Use TensorFlow's record iterator
        for record in tf.data.TFRecordDataset([filepath]):
            event = tf.compat.v1.Event.FromString(record.numpy())

            # Process summary values
            for value in event.summary.value:
                if value.HasField('simple_value'):
                    tag = value.tag
                    step = event.step
                    val = value.simple_value

                    data[tag]['steps'].append(step)
                    data[tag]['values'].append(val)
    except Exception as e:
        print(f"Error parsing {filepath}: {e}")

    return data

# Load both event files
print("Loading DQN event file...")
dqn_data = parse_tfevents_file(dqn_file)

print("Loading DDQN event file...")
ddqn_data = parse_tfevents_file(ddqn_file)

# Check what tags we found
print("\nDQN tags found:", list(dqn_data.keys()))
print("DDQN tags found:", list(ddqn_data.keys()))

# Extract data for each metric
dqn_steps_rew = dqn_data['AvgRew']['steps']
dqn_values_rew = dqn_data['AvgRew']['values']

dqn_steps_len = dqn_data['AvgEpLen']['steps']
dqn_values_len = dqn_data['AvgEpLen']['values']

dqn_steps_ep = dqn_data['Episodes']['steps']
dqn_values_ep = dqn_data['Episodes']['values']

ddqn_steps_rew = ddqn_data['AvgRew']['steps']
ddqn_values_rew = ddqn_data['AvgRew']['values']

ddqn_steps_len = ddqn_data['AvgEpLen']['steps']
ddqn_values_len = ddqn_data['AvgEpLen']['values']

ddqn_steps_ep = ddqn_data['Episodes']['steps']
ddqn_values_ep = ddqn_data['Episodes']['values']

# Create comparison visualization
fig, axes = plt.subplots(3, 1, figsize=(14, 12))

# Plot Average Reward Comparison
axes[0].plot(dqn_steps_rew, dqn_values_rew, linewidth=2, color='#2E86AB', label='DQN', alpha=0.8)
axes[0].plot(ddqn_steps_rew, ddqn_values_rew, linewidth=2, color='#F18F01', label='Double DQN', alpha=0.8)
axes[0].set_xlabel('Training Steps', fontsize=12)
axes[0].set_ylabel('Average Reward', fontsize=12)
axes[0].set_title('Fish Derby: Average Reward Comparison (DQN vs DDQN)', fontsize=14, fontweight='bold')
axes[0].legend(fontsize=11, loc='best')
axes[0].grid(True, alpha=0.3)

# Plot Average Episode Length Comparison
axes[1].plot(dqn_steps_len, dqn_values_len, linewidth=2, color='#2E86AB', label='DQN', alpha=0.8)
axes[1].plot(ddqn_steps_len, ddqn_values_len, linewidth=2, color='#F18F01', label='Double DQN', alpha=0.8)
axes[1].set_xlabel('Training Steps', fontsize=12)
axes[1].set_ylabel('Average Episode Length', fontsize=12)
axes[1].set_title('Fish Derby: Average Episode Length Comparison (DQN vs DDQN)', fontsize=14, fontweight='bold')
axes[1].legend(fontsize=11, loc='best')
axes[1].grid(True, alpha=0.3)

# Plot Total Episodes Comparison
axes[2].plot(dqn_steps_ep, dqn_values_ep, linewidth=2, color='#2E86AB', label='DQN', alpha=0.8)
axes[2].plot(ddqn_steps_ep, ddqn_values_ep, linewidth=2, color='#F18F01', label='Double DQN', alpha=0.8)
axes[2].set_xlabel('Training Steps', fontsize=12)
axes[2].set_ylabel('Total Episodes', fontsize=12)
axes[2].set_title('Fish Derby: Total Episodes Completed', fontsize=14, fontweight='bold')
axes[2].legend(fontsize=11, loc='best')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('fish_derby_comparison.png', dpi=300, bbox_inches='tight')
print("\nVisualization saved as 'fish_derby_comparison.png'")

# Print summary statistics
print(f"\n{'='*60}")
print(f"{'FISH DERBY TRAINING COMPARISON: DQN vs DDQN':^60}")
print(f"{'='*60}\n")

print(f"{'Metric':<30} {'DQN':>15} {'Double DQN':>15}")
print(f"{'-'*60}")
print(f"{'Total Steps':<30} {dqn_steps_rew[-1]:>15,} {ddqn_steps_rew[-1]:>15,}")
print(f"{'Total Episodes':<30} {int(dqn_values_ep[-1]):>15,} {int(ddqn_values_ep[-1]):>15,}")
print(f"{'-'*60}")
print(f"{'Final Avg Reward':<30} {dqn_values_rew[-1]:>15.2f} {ddqn_values_rew[-1]:>15.2f}")
print(f"{'Max Avg Reward':<30} {max(dqn_values_rew):>15.2f} {max(ddqn_values_rew):>15.2f}")
print(f"{'Final Avg Episode Length':<30} {dqn_values_len[-1]:>15.2f} {ddqn_values_len[-1]:>15.2f}")
print(f"{'-'*60}")

# Calculate performance improvement
final_reward_improvement = ((ddqn_values_rew[-1] - dqn_values_rew[-1]) / abs(dqn_values_rew[-1])) * 100
max_reward_improvement = ((max(ddqn_values_rew) - max(dqn_values_rew)) / abs(max(dqn_values_rew))) * 100

print(f"\n{'PERFORMANCE IMPROVEMENT (DDQN vs DQN)':^60}")
print(f"{'-'*60}")
print(f"{'Final Avg Reward':<30} {final_reward_improvement:>15.1f}%")
print(f"{'Max Avg Reward':<30} {max_reward_improvement:>15.1f}%")
print(f"{'='*60}\n")
