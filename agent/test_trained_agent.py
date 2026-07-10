import torch
import numpy as np
import sys
import os
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.network import DQNetwork, DuelingDQNetwork
from env.walk_env import HumanoidWalkEnv


def test_agent(model_path, num_episodes=5, render=True, initial_pose=None):
    """
    Test a trained DQN agent.
    
    Args:
        model_path: Path to saved model checkpoint
        num_episodes: Number of episodes to test
        render: Whether to render the environment
        initial_pose: Initial pose vector (if None, uses default)
    """
    
    # Create environment
    render_mode = "human" if render else None
    env = HumanoidWalkEnv(render_mode=render_mode)
    
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Try to determine network type from checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Create network (try Dueling first, fallback to standard)
    try:
        model = DuelingDQNetwork(
            state_dim=env.observation_space.shape[0],
            num_joints=env.num_joints,
            num_bins=env.num_bins
        ).to(device)
        model.load_state_dict(checkpoint['policy_net'])
        print("Loaded Dueling DQN model")
    except:
        model = DQNetwork(
            state_dim=env.observation_space.shape[0],
            num_joints=env.num_joints,
            num_bins=env.num_bins
        ).to(device)
        model.load_state_dict(checkpoint['policy_net'])
        print("Loaded standard DQN model")
    
    model.eval()
    
    print(f"\nTesting model: {model_path}")
    print(f"Episodes: {num_episodes}\n")
    print("="*60)
    
    episode_rewards = []
    episode_lengths = []
    
    for episode in range(num_episodes):
        state, _ = env.reset(initial_pose=initial_pose)
        episode_reward = 0
        step = 0
        
        while True:
            # Select action (greedy, no exploration)
            action = model.get_action(state, epsilon=0.0)
            
            # Take step
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            state = next_state
            step += 1
            
            if done:
                break
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(step)
        
        print(f"Episode {episode + 1}:")
        print(f"  Total Reward: {episode_reward:.2f}")
        print(f"  Episode Length: {step}")
        print(f"  Reason: {'Terminated (fell)' if terminated else 'Truncated (time limit)'}")
        if 'reward_components' in info:
            print(f"  Final Forward Velocity: {info['reward_components']['forward_velocity']:.3f} m/s")
        print()
    
    print("="*60)
    print(f"Average Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Average Length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
    print("="*60)
    
    env.close()
    
    return episode_rewards, episode_lengths


def test_with_pose_from_image(model_path, image_path):
    """
    Test agent with initial pose extracted from an image.
    
    Args:
        model_path: Path to trained model
        image_path: Path to image file
    """
    # Import pose extraction modules
    try:
        from pose.keypoints import extract_keypoints
        from pose.angles import skeleton_to_pose_vector
    except ImportError:
        print("Error: Module 1 (pose extraction) not available")
        return
    
    print(f"Extracting pose from image: {image_path}")
    
    # Extract pose
    keypoints = extract_keypoints(image_path)
    theta_init = skeleton_to_pose_vector(keypoints)
    
    print(f"Extracted pose vector: {theta_init}")
    print(f"Pose dimension: {len(theta_init)}")
    
    # Pad or truncate to match 8 joints if necessary
    if len(theta_init) < 8:
        theta_init = np.pad(theta_init, (0, 8 - len(theta_init)), 'constant')
        print(f"Padded pose to 8 joints: {theta_init}")
    elif len(theta_init) > 8:
        theta_init = theta_init[:8]
        print(f"Truncated pose to 8 joints: {theta_init}")
    
    # Test with this pose
    test_agent(model_path, num_episodes=3, render=True, initial_pose=theta_init)


def compare_poses(model_path):
    """
    Compare agent performance across different initial poses.
    
    Args:
        model_path: Path to trained model
    """
    poses = {
        "Standing": np.zeros(8),
        "Leaning Forward": np.array([0.5, 1.0, -0.3, -0.5, -1.0, 0.3, 0.2, -0.2]),
        "Leaning Back": np.array([-0.5, 1.0, -0.3, 0.5, -1.0, 0.3, -0.2, 0.2]),
        "Left Leg Forward": np.array([0.8, 1.2, -0.5, -0.2, -0.5, 0.2, 0.1, -0.1]),
        "Right Leg Forward": np.array([-0.2, -0.5, 0.2, 0.8, 1.2, -0.5, 0.1, -0.1]),
    }
    
    print("\n" + "="*60)
    print("Comparing Performance Across Different Initial Poses")
    print("="*60 + "\n")
    
    results = {}
    
    for pose_name, pose in poses.items():
        print(f"\nTesting pose: {pose_name}")
        print("-" * 40)
        
        rewards, lengths = test_agent(
            model_path,
            num_episodes=3,
            render=False,
            initial_pose=pose
        )
        
        results[pose_name] = {
            'avg_reward': np.mean(rewards),
            'avg_length': np.mean(lengths)
        }
    
    # Print summary
    print("\n" + "="*60)
    print("Summary: Performance by Initial Pose")
    print("="*60)
    print(f"{'Pose':<20} {'Avg Reward':<15} {'Avg Length':<15}")
    print("-" * 60)
    
    for pose_name, stats in results.items():
        print(f"{pose_name:<20} {stats['avg_reward']:<15.2f} {stats['avg_length']:<15.1f}")
    
    print("="*60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test trained DQN agent")
    parser.add_argument(
        "--model",
        type=str,
        default="checkpoints/dqn_final.pth",
        help="Path to trained model checkpoint"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=5,
        help="Number of test episodes"
    )
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Disable rendering"
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to image for pose extraction"
    )
    parser.add_argument(
        "--compare-poses",
        action="store_true",
        help="Compare performance across different poses"
    )
    
    args = parser.parse_args()
    
    # Check if model exists
    if not os.path.exists(args.model):
        print(f"Error: Model file not found: {args.model}")
        print("Please train a model first using train_dqn.py")
        sys.exit(1)
    
    # Run appropriate test
    if args.compare_poses:
        compare_poses(args.model)
    elif args.image:
        test_with_pose_from_image(args.model, args.image)
    else:
        test_agent(
            args.model,
            num_episodes=args.episodes,
            render=not args.no_render
        )