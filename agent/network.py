import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DQNetwork(nn.Module):
    """
    Deep Q-Network for multi-discrete action space.
    Each joint has its own output head with num_bins neurons.
    """
    
    def __init__(self, state_dim, num_joints, num_bins, hidden_sizes=[256, 256, 128]):
        """
        Initialize the Q-Network.
        
        Args:
            state_dim: Dimension of the state/observation space
            num_joints: Number of controllable joints
            num_bins: Number of discrete torque bins per joint
            hidden_sizes: List of hidden layer sizes
        """
        super(DQNetwork, self).__init__()
        
        self.state_dim = state_dim
        self.num_joints = num_joints
        self.num_bins = num_bins
        self.total_actions = num_joints * num_bins
        
        # Shared feature extraction layers
        layers = []
        prev_size = state_dim
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.LayerNorm(hidden_size))  # Add layer normalization for stability
            prev_size = hidden_size
        
        self.shared_net = nn.Sequential(*layers)
        
        # Output layer: separate head for each joint
        self.output_layer = nn.Linear(prev_size, self.total_actions)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, state):
        """
        Forward pass through the network.
        
        Args:
            state: State tensor of shape (batch_size, state_dim)
        
        Returns:
            q_values: Q-values reshaped to (batch_size, num_joints, num_bins)
        """
        # Shared feature extraction
        features = self.shared_net(state)
        
        # Output layer
        q_values = self.output_layer(features)
        
        # Reshape to (batch_size, num_joints, num_bins)
        batch_size = state.shape[0]
        q_values = q_values.view(batch_size, self.num_joints, self.num_bins)
        
        return q_values
    
    def get_action(self, state, epsilon=0.0):
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state: Current state (numpy array)
            epsilon: Exploration rate
        
        Returns:
            action: Selected action indices for each joint (numpy array)
        """
        if np.random.random() < epsilon:
            # Random action
            action = np.random.randint(0, self.num_bins, size=self.num_joints)
        else:
            # Greedy action
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                q_values = self.forward(state_tensor)
                # Select argmax for each joint
                action = torch.argmax(q_values, dim=2).squeeze(0).cpu().numpy()
        
        return action


class DuelingDQNetwork(nn.Module):
    """
    Dueling DQN architecture with separate value and advantage streams.
    Often provides better performance for locomotion tasks.
    """
    
    def __init__(self, state_dim, num_joints, num_bins, hidden_sizes=[256, 256]):
        """
        Initialize the Dueling Q-Network.
        
        Args:
            state_dim: Dimension of the state/observation space
            num_joints: Number of controllable joints
            num_bins: Number of discrete torque bins per joint
            hidden_sizes: List of hidden layer sizes for shared network
        """
        super(DuelingDQNetwork, self).__init__()
        
        self.state_dim = state_dim
        self.num_joints = num_joints
        self.num_bins = num_bins
        self.total_actions = num_joints * num_bins
        
        # Shared feature extraction
        layers = []
        prev_size = state_dim
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.LayerNorm(hidden_size))
            prev_size = hidden_size
        
        self.shared_net = nn.Sequential(*layers)
        
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(prev_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(prev_size, 128),
            nn.ReLU(),
            nn.Linear(128, self.total_actions)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, state):
        """
        Forward pass through the dueling network.
        
        Args:
            state: State tensor of shape (batch_size, state_dim)
        
        Returns:
            q_values: Q-values reshaped to (batch_size, num_joints, num_bins)
        """
        batch_size = state.shape[0]
        
        # Shared features
        features = self.shared_net(state)
        
        # Value and advantage
        value = self.value_stream(features)  # (batch_size, 1)
        advantage = self.advantage_stream(features)  # (batch_size, total_actions)
        
        # Reshape advantage
        advantage = advantage.view(batch_size, self.num_joints, self.num_bins)
        
        # Combine: Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        q_values = value.unsqueeze(2) + (advantage - advantage.mean(dim=2, keepdim=True))
        
        return q_values
    
    def get_action(self, state, epsilon=0.0):
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state: Current state (numpy array)
            epsilon: Exploration rate
        
        Returns:
            action: Selected action indices for each joint (numpy array)
        """
        if np.random.random() < epsilon:
            action = np.random.randint(0, self.num_bins, size=self.num_joints)
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                q_values = self.forward(state_tensor)
                action = torch.argmax(q_values, dim=2).squeeze(0).cpu().numpy()
        
        return action


# Test the networks
if __name__ == "__main__":
    print("Testing DQN Networks...")
    
    state_dim = 29
    num_joints = 8
    num_bins = 5
    batch_size = 32
    
    # Test standard DQN
    print("\n[1] Testing Standard DQN:")
    dqn = DQNetwork(state_dim, num_joints, num_bins)
    print(f"Network created with {sum(p.numel() for p in dqn.parameters())} parameters")
    
    # Forward pass
    dummy_state = torch.randn(batch_size, state_dim)
    q_values = dqn(dummy_state)
    print(f"Q-values shape: {q_values.shape}")  # Should be (32, 8, 5)
    
    # Get action
    single_state = np.random.randn(state_dim)
    action = dqn.get_action(single_state, epsilon=0.1)
    print(f"Action shape: {action.shape}, values: {action}")
    
    # Test Dueling DQN
    print("\n[2] Testing Dueling DQN:")
    dueling_dqn = DuelingDQNetwork(state_dim, num_joints, num_bins)
    print(f"Network created with {sum(p.numel() for p in dueling_dqn.parameters())} parameters")
    
    q_values_dueling = dueling_dqn(dummy_state)
    print(f"Q-values shape: {q_values_dueling.shape}")
    
    action_dueling = dueling_dqn.get_action(single_state, epsilon=0.0)
    print(f"Action shape: {action_dueling.shape}, values: {action_dueling}")
    
    print("\n✓ Network tests passed!")