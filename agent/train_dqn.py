import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import sys
from datetime import datetime
import json
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.network import DQNetwork, DuelingDQNetwork
from agent.replay import ReplayBuffer, PrioritizedReplayBuffer
from env.walk_env import HumanoidWalkEnv

# -----------------------------
# Pose dataset utilities
# -----------------------------
def _load_pose_vectors_from_dir(poses_dir: str, target_dim: int = 8):
    """
    Load all .npy pose vectors from a directory and normalize to target_dim.

    - Finds files ending with "_pose_vector.npy" (and any .npy fallback)
    - Pads/truncates to target_dim
    - Returns list[np.ndarray]
    """
    poses = []
    if poses_dir is None or not os.path.exists(poses_dir):
        return poses
    for root, _, files in os.walk(poses_dir):
        for fname in sorted(files):
            if not fname.endswith('.npy'):
                continue
            if not (fname.endswith('_pose_vector.npy') or fname.endswith('.npy')):
                continue
            fpath = os.path.join(root, fname)
            try:
                vec = np.load(fpath)
                # Flatten and cast
                vec = np.asarray(vec, dtype=np.float32).reshape(-1)
                # Normalize to target_dim
                if vec.shape[0] < target_dim:
                    vec = np.pad(vec, (0, target_dim - vec.shape[0]), 'constant')
                elif vec.shape[0] > target_dim:
                    vec = vec[:target_dim]
                # Clip to [-pi, pi] for safety
                vec = np.clip(vec, -np.pi, np.pi)
                poses.append(vec)
            except Exception:
                # Skip unreadable files
                continue
    return poses

def _prefill_replay_with_rollouts(env, agent, poses, num_rollouts: int = 50, steps_per_rollout: int = 50):
    """
    Run short warm-start rollouts from dataset poses to prefill replay buffer.
    Uses agent's current policy with high exploration (epsilon).
    """
    if not poses:
        return 0
    original_epsilon = agent.epsilon
    agent.epsilon = max(0.5, original_epsilon)  # encourage exploration during prefill
    added = 0
    for i in range(num_rollouts):
        pose = poses[i % len(poses)]
        state, _ = env.reset(initial_pose=pose)
        for _ in range(steps_per_rollout):
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.store_transition(state, action, reward, next_state, done)
            added += 1
            state = next_state
            if done:
                break
    agent.epsilon = original_epsilon
    return added


class DQNAgent:
    """
    DQN Agent for humanoid locomotion with support for:
    - Standard DQN and Dueling DQN
    - Target network with soft/hard updates
    - Experience replay
    - Epsilon-greedy exploration
    """
    
    def __init__(
        self,
        env,
        use_dueling=True,
        use_prioritized=False,
        learning_rate=1e-4,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        target_update_freq=10,
        soft_update_tau=0.005,
        use_soft_update=True,
        buffer_capacity=100000,
        batch_size=64,
        device=None
    ):
        """
        Initialize DQN agent.
        
        Args:
            env: Gymnasium environment
            use_dueling: Use Dueling DQN architecture
            use_prioritized: Use prioritized experience replay
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            epsilon_start: Initial exploration rate
            epsilon_end: Final exploration rate
            epsilon_decay: Epsilon decay rate
            target_update_freq: Frequency of target network updates (hard update)
            soft_update_tau: Tau for soft target updates
            use_soft_update: Use soft updates instead of hard updates
            buffer_capacity: Replay buffer capacity
            batch_size: Batch size for training
            device: Device for training (cuda/cpu)
        """
        self.env = env
        self.state_dim = env.observation_space.shape[0]
        self.num_joints = env.num_joints
        self.num_bins = env.num_bins
        self.gamma = gamma
        self.batch_size = batch_size
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update_freq = target_update_freq
        self.soft_update_tau = soft_update_tau
        self.use_soft_update = use_soft_update
        
        # Device
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Networks
        NetworkClass = DuelingDQNetwork if use_dueling else DQNetwork
        self.policy_net = NetworkClass(self.state_dim, self.num_joints, self.num_bins).to(self.device)
        self.target_net = NetworkClass(self.state_dim, self.num_joints, self.num_bins).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        
        # Replay buffer
        if use_prioritized:
            self.replay_buffer = PrioritizedReplayBuffer(capacity=buffer_capacity)
        else:
            self.replay_buffer = ReplayBuffer(capacity=buffer_capacity)
        self.use_prioritized = use_prioritized
        
        # Training stats
        self.steps_done = 0
        self.episodes_done = 0
    
    def select_action(self, state):
        """Select action using epsilon-greedy policy."""
        action = self.policy_net.get_action(state, epsilon=self.epsilon)
        return action
    
    def store_transition(self, state, action, reward, next_state, done):
        """Store transition in replay buffer."""
        self.replay_buffer.push(state, action, reward, next_state, done)
    
    def update_epsilon(self):
        """Decay epsilon for exploration."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def soft_update_target(self):
        """Soft update of target network parameters."""
        for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(
                self.soft_update_tau * policy_param.data + (1.0 - self.soft_update_tau) * target_param.data
            )
    
    def hard_update_target(self):
        """Hard update of target network parameters."""
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def train_step(self):
        """
        Perform one training step.
        
        Returns:
            loss: Training loss value
        """
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        # Sample from replay buffer
        if self.use_prioritized:
            states, actions, rewards, next_states, dones, indices, weights = \
                self.replay_buffer.sample(self.batch_size, beta=0.4)
            weights = torch.FloatTensor(weights).to(self.device)
        else:
            states, actions, rewards, next_states, dones = \
                self.replay_buffer.sample(self.batch_size)
            weights = torch.ones(self.batch_size).to(self.device)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Current Q-values
        q_values = self.policy_net(states)  # (batch, num_joints, num_bins)
        
        # Gather Q-values for taken actions
        q_values_selected = torch.gather(q_values, 2, actions.unsqueeze(2)).squeeze(2)  # (batch, num_joints)
        q_values_selected = q_values_selected.mean(dim=1)  # Average across joints
        
        # Target Q-values
        with torch.no_grad():
            next_q_values = self.target_net(next_states)
            next_q_values_max = next_q_values.max(dim=2)[0].mean(dim=1)  # Max then average
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values_max
        
        # Compute loss
        td_errors = target_q_values - q_values_selected
        loss = (weights * td_errors.pow(2)).mean()
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
        self.optimizer.step()
        
        # Update priorities if using prioritized replay
        if self.use_prioritized:
            priorities = td_errors.abs().detach().cpu().numpy()
            self.replay_buffer.update_priorities(indices, priorities)
        
        # Update target network
        if self.use_soft_update:
            self.soft_update_target()
        elif self.steps_done % self.target_update_freq == 0:
            self.hard_update_target()
        
        self.steps_done += 1
        
        return loss.item()
    
    def save(self, filepath):
        """Save agent state."""
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps_done': self.steps_done,
            'episodes_done': self.episodes_done
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath):
        """Load agent state."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.steps_done = checkpoint['steps_done']
        self.episodes_done = checkpoint['episodes_done']
        print(f"Model loaded from {filepath}")


def train(
    env,
    agent,
    num_episodes=1000,
    max_steps_per_episode=1000,
    save_freq=50,
    log_freq=10,
    save_dir="checkpoints",
    pose_library=None,
    prefill_rollouts=0,
    prefill_steps=50,
    action_repeat=1
):
    """
    Train the DQN agent.
    
    Args:
        env: Environment
        agent: DQN agent
        num_episodes: Number of training episodes
        max_steps_per_episode: Maximum steps per episode
        save_freq: Frequency of model saving
        log_freq: Frequency of logging
        save_dir: Directory to save checkpoints
        pose_library: List of initial poses for curriculum learning
    """
    os.makedirs(save_dir, exist_ok=True)
    
    episode_rewards = []
    episode_lengths = []
    losses = []
    
    print(f"\n{'='*60}")
    print(f"Starting Training: {num_episodes} episodes")
    print(f"{'='*60}\n")
    
    # Optional: prefill buffer with short rollouts
    if prefill_rollouts > 0 and pose_library is not None and len(pose_library) > 0:
        print(f"Prefilling replay buffer with {prefill_rollouts} warm-start rollouts × {prefill_steps} steps...")
        added = _prefill_replay_with_rollouts(env, agent, pose_library, prefill_rollouts, prefill_steps)
        print(f"Prefill complete: {added} transitions added. Current buffer size: {len(agent.replay_buffer)}\n")
    
    for episode in range(num_episodes):
        # Select initial pose from library if available
        if pose_library is not None and len(pose_library) > 0:
            initial_pose = pose_library[np.random.randint(len(pose_library))]
        else:
            initial_pose = None
        
        state, _ = env.reset(initial_pose=initial_pose)
        episode_reward = 0
        episode_loss = []
        
        step = 0
        while step < max_steps_per_episode:
            # Select action
            action = agent.select_action(state)

            # Repeat action for stability
            accumulated_reward = 0.0
            repeated_terminated = False
            repeated_truncated = False
            for _ in range(action_repeat):
                next_state, reward, terminated, truncated, info = env.step(action)
                done_inner = terminated or truncated
                agent.store_transition(state, action, reward, next_state, done_inner)

                # Train after each env step
                loss = agent.train_step()
                if loss is not None:
                    episode_loss.append(loss)

                accumulated_reward += reward
                state = next_state
                step += 1
                if done_inner or step >= max_steps_per_episode:
                    repeated_terminated = terminated
                    repeated_truncated = truncated
                    break

            episode_reward += accumulated_reward
            if repeated_terminated or repeated_truncated or step >= max_steps_per_episode:
                break
        
        # Update epsilon
        agent.update_epsilon()
        agent.episodes_done += 1
        
        # Log statistics
        episode_rewards.append(episode_reward)
        episode_lengths.append(step + 1)
        if episode_loss:
            losses.append(np.mean(episode_loss))
        
        # Print progress
        if (episode + 1) % log_freq == 0:
            avg_reward = np.mean(episode_rewards[-log_freq:])
            avg_length = np.mean(episode_lengths[-log_freq:])
            avg_loss = np.mean(losses[-log_freq:]) if losses else 0
            
            print(f"Episode {episode + 1}/{num_episodes}")
            print(f"  Avg Reward: {avg_reward:.2f}")
            print(f"  Avg Length: {avg_length:.1f}")
            print(f"  Avg Loss: {avg_loss:.4f}")
            print(f"  Epsilon: {agent.epsilon:.4f}")
            print(f"  Buffer Size: {len(agent.replay_buffer)}")
            print()
        
        # Save checkpoint
        if (episode + 1) % save_freq == 0:
            save_path = os.path.join(save_dir, f"dqn_episode_{episode + 1}.pth")
            agent.save(save_path)
            
            # Save training stats
            stats = {
                'episode_rewards': episode_rewards,
                'episode_lengths': episode_lengths,
                'losses': losses
            }
            stats_path = os.path.join(save_dir, f"training_stats_{episode + 1}.json")
            with open(stats_path, 'w') as f:
                json.dump(stats, f)
    
    print(f"\n{'='*60}")
    print("Training Complete!")
    print(f"{'='*60}\n")
    
    return episode_rewards, episode_lengths, losses


# Main training script
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train DQN with optional pose dataset integration and action repeat")
    parser.add_argument("--pose-dir", type=str, default="data/output_image", help="Directory containing *_pose_vector.npy files")
    parser.add_argument("--episodes", type=int, default=150, help="Number of training episodes")
    parser.add_argument("--max-steps", type=int, default=2000, help="Max steps per episode")
    parser.add_argument("--save-freq", type=int, default=30, help="Checkpoint save frequency (episodes)")
    parser.add_argument("--log-freq", type=int, default=5, help="Logging frequency (episodes)")
    parser.add_argument("--prefill-rollouts", type=int, default=300, help="Number of warm-start rollouts to prefill replay buffer")
    parser.add_argument("--action-repeat", type=int, default=1, help="Repeat each action this many env steps for stability")
    parser.add_argument("--prefill-steps", type=int, default=50, help="Steps per warm-start rollout")
    parser.add_argument("--dueling", action="store_true", help="Use Dueling DQN")
    parser.add_argument("--prioritized", action="store_true", help="Use prioritized replay")
    parser.add_argument("--save-dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    args = parser.parse_args()

    print("Initializing DQN Training...")
    
    # Create environment
    env = HumanoidWalkEnv(render_mode=None, max_episode_steps=args.max_steps)
    
    # Create agent
    agent = DQNAgent(
        env=env,
        use_dueling=args.dueling or True,
        use_prioritized=args.prioritized or False,
        learning_rate=1e-4,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.985,
        buffer_capacity=30000,
        batch_size=32
    )
    
    # Load pose vectors from dataset directory, sized to env joints
    pose_library = _load_pose_vectors_from_dir(args.pose_dir, target_dim=env.num_joints)
    if pose_library:
        print(f"Loaded {len(pose_library)} pose vectors from {args.pose_dir}")
    else:
        print(f"No pose vectors found in {args.pose_dir}; proceeding without dataset poses")
    
    # Train
    rewards, lengths, losses = train(
        env=env,
        agent=agent,
        num_episodes=args.episodes,
        max_steps_per_episode=args.max_steps,
        save_freq=args.save_freq,
        log_freq=args.log_freq,
        save_dir=args.save_dir,
        pose_library=pose_library if pose_library else None,
        prefill_rollouts=args.prefill_rollouts if pose_library else 0,
        prefill_steps=args.prefill_steps,
        action_repeat=args.action_repeat
    )
    
    # Save final model
    agent.save("checkpoints/dqn_final.pth")
    
    # Close environment
    env.close()
    print("Training completed!")
    import os
import json
import matplotlib.pyplot as plt
import numpy as np

# Directory containing saved training stats
save_dir = "checkpoints"

# Collect all JSON stats files
json_files = sorted(
    [f for f in os.listdir(save_dir) if f.startswith("training_stats_") and f.endswith(".json")],
    key=lambda x: int(x.split("_")[-1].split(".")[0])  # sort by episode number
)

if not json_files:
    print("No training_stats_*.json files found in 'checkpoints' folder.")
    exit()

# Combine data from all files
all_rewards, all_lengths, all_losses = [], [], []
for file in json_files:
    path = os.path.join(save_dir, file)
    with open(path, "r") as f:
        stats = json.load(f)
        all_rewards.extend(stats["episode_rewards"])
        all_lengths.extend(stats["episode_lengths"])
        all_losses.extend(stats["losses"])

# Convert to numpy for smoothing
rewards = np.array(all_rewards)
lengths = np.array(all_lengths)
losses = np.array(all_losses)

# ---  Reward Curve ---
plt.figure(figsize=(10, 5))
plt.plot(rewards, label="Episode Reward", color='blue')
plt.title("Reward Curve over Episodes")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.grid(True)
plt.legend()
plt.savefig(os.path.join(save_dir, "reward_curve.png"))
plt.show()

# ---  Loss Curve ---
if len(losses) > 0:
    plt.figure(figsize=(10, 5))
    plt.plot(losses, label="Average Loss", color='red')
    plt.title("Loss Curve over Episodes")
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(save_dir, "loss_curve.png"))
    plt.show()

# ---  Episode Length Curve ---
plt.figure(figsize=(10, 5))
plt.plot(lengths, label="Episode Length", color='green')
plt.title("Episode Length over Episodes")
plt.xlabel("Episode")
plt.ylabel("Length")
plt.grid(True)
plt.legend()
plt.savefig(os.path.join(save_dir, "length_curve.png"))
plt.show()

print("\n All plots saved to:", os.path.abspath(save_dir))
