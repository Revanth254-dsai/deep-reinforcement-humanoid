import numpy as np
import random
from collections import deque


class ReplayBuffer:
    """
    Experience Replay Buffer for DQN training.
    Stores transitions and samples random batches for training.
    """
    
    def __init__(self, capacity=100000):
        """
        Initialize replay buffer.
        
        Args:
            capacity: Maximum number of transitions to store
        """
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity
    
    def push(self, state, action, reward, next_state, done):
        """
        Add a transition to the buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode terminated
        """
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """
        Sample a random batch of transitions.
        
        Args:
            batch_size: Number of transitions to sample
        
        Returns:
            Tuple of (states, actions, rewards, next_states, dones)
        """
        batch = random.sample(self.buffer, batch_size)
        
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32)
        )
    
    def __len__(self):
        """Return current size of buffer."""
        return len(self.buffer)


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay Buffer.
    Samples transitions based on their TD error for more efficient learning.
    """
    
    def __init__(self, capacity=100000, alpha=0.6):
        """
        Initialize prioritized replay buffer.
        
        Args:
            capacity: Maximum number of transitions
            alpha: Prioritization exponent (0 = uniform, 1 = full prioritization)
        """
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.size = 0
    
    def push(self, state, action, reward, next_state, done):
        """Add transition with maximum priority."""
        max_priority = self.priorities.max() if self.size > 0 else 1.0
        
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.position] = (state, action, reward, next_state, done)
        
        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size, beta=0.4):
        """
        Sample batch with priorities.
        
        Args:
            batch_size: Number of transitions
            beta: Importance sampling exponent (0 = no correction, 1 = full correction)
        
        Returns:
            Tuple of (states, actions, rewards, next_states, dones, indices, weights)
        """
        if self.size < batch_size:
            batch_size = self.size
        
        # Calculate sampling probabilities
        priorities = self.priorities[:self.size]
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()
        
        # Sample indices
        indices = np.random.choice(self.size, batch_size, p=probabilities, replace=False)
        
        # Calculate importance sampling weights
        weights = (self.size * probabilities[indices]) ** (-beta)
        weights /= weights.max()
        
        # Get transitions
        batch = [self.buffer[idx] for idx in indices]
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
            indices,
            np.array(weights, dtype=np.float32)
        )
    
    def update_priorities(self, indices, priorities):
        """
        Update priorities of sampled transitions.
        
        Args:
            indices: Indices of sampled transitions
            priorities: New priority values (TD errors)
        """
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority + 1e-5  # Add small constant to avoid zero priority
    
    def __len__(self):
        """Return current size of buffer."""
        return self.size


# Test the replay buffers
if __name__ == "__main__":
    print("Testing Replay Buffers...")
    
    # Test standard replay buffer
    print("\n[1] Testing Standard Replay Buffer:")
    buffer = ReplayBuffer(capacity=1000)
    
    # Add some transitions
    for i in range(100):
        state = np.random.randn(29)
        action = np.random.randint(0, 5, size=8)
        reward = np.random.randn()
        next_state = np.random.randn(29)
        done = False
        buffer.push(state, action, reward, next_state, done)
    
    print(f"Buffer size: {len(buffer)}")
    
    # Sample batch
    states, actions, rewards, next_states, dones = buffer.sample(32)
    print(f"Sampled batch shapes:")
    print(f"  States: {states.shape}")
    print(f"  Actions: {actions.shape}")
    print(f"  Rewards: {rewards.shape}")
    print(f"  Next states: {next_states.shape}")
    print(f"  Dones: {dones.shape}")
    
    # Test prioritized replay buffer
    print("\n[2] Testing Prioritized Replay Buffer:")
    pri_buffer = PrioritizedReplayBuffer(capacity=1000)
    
    # Add transitions
    for i in range(100):
        state = np.random.randn(29)
        action = np.random.randint(0, 5, size=8)
        reward = np.random.randn()
        next_state = np.random.randn(29)
        done = False
        pri_buffer.push(state, action, reward, next_state, done)
    
    print(f"Prioritized buffer size: {len(pri_buffer)}")
    
    # Sample with priorities
    states, actions, rewards, next_states, dones, indices, weights = pri_buffer.sample(32, beta=0.4)
    print(f"Sampled batch shapes:")
    print(f"  States: {states.shape}")
    print(f"  Indices: {indices.shape}")
    print(f"  Weights: {weights.shape}, min={weights.min():.3f}, max={weights.max():.3f}")
    
    # Update priorities
    new_priorities = np.random.rand(32) * 10
    pri_buffer.update_priorities(indices, new_priorities)
    print(f"Priorities updated")
    
    print("\n✓ Replay buffer tests passed!")