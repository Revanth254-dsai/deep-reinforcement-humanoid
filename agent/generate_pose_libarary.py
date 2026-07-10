import numpy as np
import os
import sys
import pickle
import argparse
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pose.keypoints import extract_keypoints
from pose.angles import skeleton_to_pose_vector


def generate_pose_library(image_dir, output_path="data/poses/pose_library.pkl", target_joints=8):
    """
    Generate a library of initial poses from multiple images.
    
    Args:
        image_dir: Directory containing images
        output_path: Path to save the pose library
        target_joints: Number of joints to extract (pad or truncate)
    
    Returns:
        pose_library: List of pose vectors
    """
    image_dir = Path(image_dir)
    
    # Supported image extensions
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    
    # Find all images
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(list(image_dir.glob(f"*{ext}")))
        image_paths.extend(list(image_dir.glob(f"*{ext.upper()}")))
    
    if not image_paths:
        print(f"No images found in {image_dir}")
        return []
    
    print(f"Found {len(image_paths)} images")
    print(f"Processing images to extract poses...\n")
    
    pose_library = []
    failed_images = []
    
    for i, img_path in enumerate(image_paths):
        try:
            print(f"[{i+1}/{len(image_paths)}] Processing: {img_path.name}")
            
            # Extract keypoints
            keypoints = extract_keypoints(str(img_path))
            
            # Convert to pose vector
            theta = skeleton_to_pose_vector(keypoints)
            
            # Adjust to target number of joints
            if len(theta) < target_joints:
                # Pad with zeros
                theta = np.pad(theta, (0, target_joints - len(theta)), 'constant')
            elif len(theta) > target_joints:
                # Truncate
                theta = theta[:target_joints]
            
            pose_library.append(theta)
            print(f"  ✓ Extracted pose: {theta}")
            
        except Exception as e:
            print(f"  ✗ Failed: {str(e)}")
            failed_images.append(str(img_path))
    
    print(f"\n{'='*60}")
    print(f"Pose Extraction Summary")
    print(f"{'='*60}")
    print(f"Total images processed: {len(image_paths)}")
    print(f"Successful extractions: {len(pose_library)}")
    print(f"Failed extractions: {len(failed_images)}")
    
    if failed_images:
        print(f"\nFailed images:")
        for img in failed_images:
            print(f"  - {img}")
    
    # Save pose library
    if pose_library:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'wb') as f:
            pickle.dump(pose_library, f)
        
        print(f"\n Pose library saved to: {output_path}")
        print(f"  Number of poses: {len(pose_library)}")
        
        # Also save as numpy array for easy loading
        numpy_path = output_path.replace('.pkl', '.npy')
        np.save(numpy_path, np.array(pose_library))
        print(f" Also saved as numpy array: {numpy_path}")
    else:
        print("\n No poses extracted. Library not saved.")
    
    return pose_library


def load_pose_library(library_path):
    """
    Load a saved pose library.
    
    Args:
        library_path: Path to saved pose library (.pkl or .npy)
    
    Returns:
        pose_library: List or array of pose vectors
    """
    if library_path.endswith('.pkl'):
        with open(library_path, 'rb') as f:
            return pickle.load(f)
    elif library_path.endswith('.npy'):
        return np.load(library_path).tolist()
    else:
        raise ValueError("Library must be .pkl or .npy file")


def visualize_pose_library(library_path):
    """
    Print statistics about a pose library.
    
    Args:
        library_path: Path to pose library
    """
    poses = load_pose_library(library_path)
    poses_array = np.array(poses)
    
    print(f"\n{'='*60}")
    print(f"Pose Library Statistics")
    print(f"{'='*60}")
    print(f"Library path: {library_path}")
    print(f"Number of poses: {len(poses)}")
    print(f"Pose dimension: {poses_array.shape[1]}")
    print(f"\nJoint angle statistics (in radians):")
    print(f"{'Joint':<10} {'Mean':<12} {'Std':<12} {'Min':<12} {'Max':<12}")
    print("-" * 60)
    
    for joint_idx in range(poses_array.shape[1]):
        joint_angles = poses_array[:, joint_idx]
        print(f"Joint {joint_idx:<3} "
              f"{np.mean(joint_angles):<12.3f} "
              f"{np.std(joint_angles):<12.3f} "
              f"{np.min(joint_angles):<12.3f} "
              f"{np.max(joint_angles):<12.3f}")
    
    print("="*60 + "\n")
    
    # Show first few poses
    print("Sample poses (first 5):")
    for i, pose in enumerate(poses[:5]):
        print(f"  Pose {i+1}: {pose}")
    
    print()


def add_synthetic_poses(pose_library, num_synthetic=10, noise_std=0.3):
    """
    Add synthetic poses by perturbing existing poses.
    
    Args:
        pose_library: Existing pose library
        num_synthetic: Number of synthetic poses to add
        noise_std: Standard deviation of Gaussian noise
    
    Returns:
        augmented_library: Library with synthetic poses added
    """
    if not pose_library:
        print("Empty pose library, cannot generate synthetic poses")
        return pose_library
    
    poses_array = np.array(pose_library)
    synthetic_poses = []
    
    for i in range(num_synthetic):
        # Select random base pose
        base_pose = poses_array[np.random.randint(len(poses_array))]
        
        # Add Gaussian noise
        noise = np.random.normal(0, noise_std, size=base_pose.shape)
        synthetic_pose = base_pose + noise
        
        # Clip to reasonable range [-π, π]
        synthetic_pose = np.clip(synthetic_pose, -np.pi, np.pi)
        
        synthetic_poses.append(synthetic_pose)
    
    augmented_library = pose_library + synthetic_poses
    
    print(f"Added {num_synthetic} synthetic poses")
    print(f"Total poses in library: {len(augmented_library)}")
    
    return augmented_library


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate pose library from images")
    parser.add_argument(
        "--image-dir",
        type=str,
        default="data/images",
        help="Directory containing images"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/poses/pose_library.pkl",
        help="Output path for pose library"
    )
    parser.add_argument(
        "--joints",
        type=int,
        default=8,
        help="Number of joints to extract"
    )
    parser.add_argument(
        "--visualize",
        type=str,
        default=None,
        help="Path to existing library to visualize"
    )
    parser.add_argument(
        "--add-synthetic",
        type=int,
        default=0,
        help="Number of synthetic poses to add"
    )
    
    args = parser.parse_args()
    
    # Visualize existing library
    if args.visualize:
        visualize_pose_library(args.visualize)
    else:
        # Generate new library
        pose_library = generate_pose_library(
            args.image_dir,
            args.output,
            args.joints
        )
        
        # Add synthetic poses if requested
        if args.add_synthetic > 0 and pose_library:
            pose_library = add_synthetic_poses(pose_library, args.add_synthetic)
            
            # Save augmented library
            with open(args.output, 'wb') as f:
                pickle.dump(pose_library, f)
            
            numpy_path = args.output.replace('.pkl', '.npy')
            np.save(numpy_path, np.array(pose_library))
            
            print(f"\n✓ Augmented library saved")
        
        # Show statistics
        if pose_library:
            visualize_pose_library(args.output)