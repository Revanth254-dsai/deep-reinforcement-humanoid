import sys
import os
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from env.walk_env import HumanoidWalkEnv
from agent.train_dqn import DQNAgent, train

print("="*70)
print("HYPERPARAMETER TUNING for Better Walking")
print("="*70)

# Create environment with modified rewards
env = HumanoidWalkEnv(render_mode=None)

# Override reward weights
env.w_vel = 3.0      # Increase forward velocity reward
env.w_live = 0.05    # Reduce alive bonus
env.w_energy = 0.005 # Increase energy penalty (encourage efficiency)

print("\nModified Reward Function:")
print(f"  Forward velocity weight: {env.w_vel}")
print(f"  Alive bonus weight: {env.w_live}")
print(f"  Energy penalty weight: {env.w_energy}\n")

# Create agent with better hyperparameters
agent = DQNAgent(
    env=env,
    use_dueling=True,
    learning_rate=3e-4,  # Higher learning rate
    gamma=0.99,
    epsilon_start=1.0,
    epsilon_end=0.01,    # Lower end epsilon
    epsilon_decay=0.99,  # Slower decay
    buffer_capacity=50000,  # Larger buffer
    batch_size=64,       # Larger batch
    target_update_freq=10,
    use_soft_update=True,
    soft_update_tau=0.01  # Faster target updates
)

print("Optimized Hyperparameters:")
print(f"  Learning rate: 3e-4")
print(f"  Batch size: 64")
print(f"  Buffer: 50,000")
print(f"  Epsilon decay: 0.99\n")

# Pose library
pose_library = [
    np.zeros(8),
    np.array([0.3, 0.5, -0.2, -0.3, -0.5, 0.2, 0.1, -0.1]),
    np.array([-0.2, 0.8, -0.3, 0.2, -0.8, 0.3, -0.1, 0.1]),
    np.array([0.5, 1.0, -0.3, -0.5, -1.0, 0.3, 0.2, -0.2]),
    np.array([-0.5, 1.0, -0.3, 0.5, -1.0, 0.3, -0.2, 0.2]),
]

print("Starting optimized training (200 episodes)...\n")

rewards, lengths, losses = train(
    env=env,
    agent=agent,
    num_episodes=200,
    max_steps_per_episode=500,
    save_freq=40,
    log_freq=10,
    save_dir="checkpoints_tuned",
    pose_library=pose_library
)

agent.save("checkpoints_tuned/dqn_optimized.pth")

print("\n" + "="*70)
print("OPTIMIZED TRAINING COMPLETE")
print("="*70)
print(f"Average reward (last 10): {np.mean(rewards[-10:]):.2f}")
print(f"Model: checkpoints_tuned/dqn_optimized.pth")
print("="*70)

env.close()