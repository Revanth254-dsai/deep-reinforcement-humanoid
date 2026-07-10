import numpy as np

# BODY_25 simplified limb connections (parent -> child)
JOINT_PAIRS = {
    "left_leg":  [(9, 10), (10, 11)],       # RHip->RKnee, RKnee->RAnkle
    "right_leg": [(12, 13), (13, 14)],      # LHip->LKnee, LKnee->LAnkle
    "left_arm":  [(5, 6), (6, 7)],          # LShoulder->LElbow->LWrist
    "right_arm": [(2, 3), (3, 4)],          # RShoulder->RElbow->RWrist
    "spine":     [(1, 8)],                  # Neck->MidHip
}

# BODY_25 keypoint names
BODY_25_NAMES = [
    "Nose", "Neck", "RShoulder", "RElbow", "RWrist",
    "LShoulder", "LElbow", "LWrist", "MidHip",
    "RHip", "RKnee", "RAnkle", "LHip", "LKnee",
    "LAnkle", "REye", "LEye", "REar", "LEar",
    "LBigToe", "LSmallToe", "LHeel",
    "RBigToe", "RSmallToe", "RHeel"
]


def skeleton_to_pose_vector(keypoints_all: np.ndarray, selected_index=0, verbose=True):
    """
    Convert skeleton to pose vector with joint angles for SELECTED PERSON.
    Shows detailed angle calculation for each joint.
    
    Args:
        keypoints_all: Array of shape (num_people, 25, 3) or (25, 3)
        selected_index: Index of person to use
        verbose: Print detailed information
    
    Returns:
        theta_init: 1D array of joint angles (radians)
        metadata: Dictionary with processing metadata
    """
    
    # Handle single-person vs multi-person input
    if keypoints_all.ndim == 2:
        keypoints_all = np.expand_dims(keypoints_all, 0)
    
    best_skeleton = keypoints_all[selected_index]
    
    if verbose:
        print(f"{'='*120}")
        print(f"STEP 4: CALCULATE JOINT ANGLES (Person {selected_index + 1})")
        print(f"{'='*120}\n")
    
    # Calculate joint angles
    angles = []
    angle_details = []
    
    if verbose:
        print(f"{'Limb':<15} {'Joint Pair':<30} {'Point_A (x,y)':<20} {'Point_B (x,y)':<20} {'Vector (dx,dy)':<25} {'Angle (rad)':<15} {'Angle (deg)':<15}")
        print(f"{'-'*120}\n")
    
    joint_counter = 1
    
    for limb_name, pairs in JOINT_PAIRS.items():
        for (joint_a, joint_b) in pairs:
            # Get keypoint coordinates
            point_a = best_skeleton[joint_a]
            point_b = best_skeleton[joint_b]
            
            joint_a_name = BODY_25_NAMES[joint_a]
            joint_b_name = BODY_25_NAMES[joint_b]
            
            # Check if both points are valid
            if point_a[2] > 0.1 and point_b[2] > 0.1:
                # Define vector between joints
                v = point_b[:2] - point_a[:2]  # [dx, dy]
                
                # Calculate angle using atan2
                angle = np.arctan2(v[1], v[0])  # radians
                angle_deg = np.degrees(angle)
                
                if verbose:
                    joint_pair_str = f"{joint_a_name} -> {joint_b_name}"
                    point_a_str = f"({point_a[0]:.1f}, {point_a[1]:.1f})"
                    point_b_str = f"({point_b[0]:.1f}, {point_b[1]:.1f})"
                    vector_str = f"({v[0]:+.2f}, {v[1]:+.2f})"
                    
                    print(f"Joint {joint_counter} | {limb_name:<14} | {joint_pair_str:<29} | {point_a_str:<19} | {point_b_str:<19} | {vector_str:<24} | {angle:+.4f}     | {angle_deg:+8.2f}deg")
                
                angles.append(angle)
                angle_details.append({
                    'joint_number': joint_counter,
                    'limb': limb_name,
                    'joint_a': joint_a_name,
                    'joint_b': joint_b_name,
                    'point_a': [float(point_a[0]), float(point_a[1]), float(point_a[2])],
                    'point_b': [float(point_b[0]), float(point_b[1]), float(point_b[2])],
                    'vector': [float(v[0]), float(v[1])],
                    'angle_rad': float(angle),
                    'angle_deg': float(angle_deg)
                })
            else:
                # Invalid keypoints - use neutral angle
                if verbose:
                    joint_pair_str = f"{joint_a_name} -> {joint_b_name}"
                    status_str = "INVALID KEYPOINTS"
                    
                    print(f"Joint {joint_counter} | {limb_name:<14} | {joint_pair_str:<29} | {'N/A':<19} | {'N/A':<19} | {'N/A':<24} | {'N/A':<11} | {status_str}")
                
                angles.append(0.0)
                angle_details.append({
                    'joint_number': joint_counter,
                    'limb': limb_name,
                    'joint_a': joint_a_name,
                    'joint_b': joint_b_name,
                    'angle_rad': 0.0,
                    'angle_deg': 0.0,
                    'status': 'invalid'
                })
            
            joint_counter += 1
    
    # Convert to numpy array
    theta_init = np.array(angles, dtype=np.float32)
    
    if verbose:
        print(f"\n{'-'*120}\n")
        print(f"{'='*120}")
        print(f"FINAL OUTPUT: Joint Angles Summary (theta_init)")
        print(f"{'='*120}")
        print(f"Type: {type(theta_init).__name__}")
        print(f"Shape: {theta_init.shape}")
        print(f"Dtype: {theta_init.dtype}")
        print(f"Total Joints: {len(theta_init)}\n")
        
        print(f"All Joint Angles (in radians and degrees):")
        print(f"{'Joint':<8} {'Radians':<15} {'Degrees':<15} {'Limb':<15} {'Joint Pair':<30}")
        print(f"{'-'*85}")
        for i, angle in enumerate(theta_init):
            if i < len(angle_details):
                detail = angle_details[i]
                limb = detail.get('limb', 'N/A')
                pair = f"{detail.get('joint_a', 'N/A')} -> {detail.get('joint_b', 'N/A')}"
                print(f"{i+1:<8} {angle:+.4f}      {np.degrees(angle):+8.2f}deg  {limb:<15} {pair:<30}")
        
        print(f"\n{'-'*85}")
        print(f"Statistics:")
        print(f"  Min angle: {np.min(theta_init):+.4f} rad ({np.degrees(np.min(theta_init)):+.2f}deg)")
        print(f"  Max angle: {np.max(theta_init):+.4f} rad ({np.degrees(np.max(theta_init)):+.2f}deg)")
        print(f"  Mean angle: {np.mean(theta_init):+.4f} rad ({np.degrees(np.mean(theta_init)):+.2f}deg)")
        print(f"  Std dev: {np.std(theta_init):+.4f} rad ({np.degrees(np.std(theta_init)):+.2f}deg)")
        print(f"{'='*120}\n")
    
    # Return metadata
    metadata = {
        'num_joints': len(theta_init),
        'angle_details': angle_details
    }
    
    return theta_init, metadata


if __name__ == "__main__":
    print("OK angles.py module ready")