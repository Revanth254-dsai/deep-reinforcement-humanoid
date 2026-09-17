"""
Evaluate a trained DQN/Dueling-DQN humanoid walking policy.

Supports:

- normal evaluation
- image-defined initial pose
- comparison across several initial poses

Image extraction uses the Python 3.10 -> Python 3.7 OpenPose bridge.
"""

import argparse
import os
import sys

import numpy as np
import torch


PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(
        0,
        PROJECT_ROOT,
    )


from agent.network import DQNetwork, DuelingDQNetwork
from env.walk_env import HumanoidWalkEnv
from pose.openpose_bridge import extract_pose_with_openpose


def load_model(
    model_path,
    env,
    device,
):
    """
    Load a Dueling DQN checkpoint if possible,
    otherwise fall back to the standard DQN network.
    """

    checkpoint = torch.load(
        model_path,
        map_location=device,
    )

    try:

        model = DuelingDQNetwork(
            state_dim=(
                env.observation_space.shape[
                    0
                ]
            ),
            num_joints=env.num_joints,
            num_bins=env.num_bins,
        ).to(
            device
        )

        model.load_state_dict(
            checkpoint["policy_net"]
        )

        print(
            "Loaded Dueling DQN model"
        )

    except Exception:

        model = DQNetwork(
            state_dim=(
                env.observation_space.shape[
                    0
                ]
            ),
            num_joints=env.num_joints,
            num_bins=env.num_bins,
        ).to(
            device
        )

        model.load_state_dict(
            checkpoint["policy_net"]
        )

        print(
            "Loaded standard DQN model"
        )

    model.eval()

    return model


def test_agent(
    model_path,
    num_episodes=5,
    render=True,
    initial_pose=None,
):
    """
    Test a trained DQN agent.

    Parameters
    ----------
    model_path : str
        Path to checkpoint.

    num_episodes : int
        Number of evaluation episodes.

    render : bool
        Enable PyBullet GUI.

    initial_pose : array-like or None
        Optional 8-joint initial pose.
    """

    render_mode = (
        "human"
        if render
        else None
    )

    env = HumanoidWalkEnv(
        render_mode=render_mode
    )

    if initial_pose is not None:

        initial_pose = np.asarray(
            initial_pose,
            dtype=np.float32,
        )

        if initial_pose.shape != (
            env.num_joints,
        ):
            env.close()

            raise ValueError(
                "Initial pose must have shape "
                f"({env.num_joints},), "
                f"received {initial_pose.shape}"
            )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model = load_model(
        model_path=model_path,
        env=env,
        device=device,
    )

    print()
    print(
        "Testing model:",
        model_path,
    )

    print(
        "Device:",
        device,
    )

    print(
        "Episodes:",
        num_episodes,
    )

    print("=" * 70)

    episode_rewards = []
    episode_lengths = []

    for episode in range(
        num_episodes
    ):

        state, reset_info = env.reset(
            initial_pose=initial_pose
        )

        if initial_pose is not None:

            applied_pose = np.asarray(
                reset_info[
                    "initial_pose"
                ],
                dtype=np.float32,
            )

            if not np.allclose(
                initial_pose,
                applied_pose,
                atol=1e-5,
            ):

                print(
                    "[NOTE] Initial pose was "
                    "clipped to URDF limits."
                )

                print(
                    "Requested:",
                    initial_pose,
                )

                print(
                    "Applied:",
                    applied_pose,
                )

        episode_reward = 0.0
        step = 0

        terminated = False
        truncated = False
        info = {}

        while True:

            action = model.get_action(
                state,
                epsilon=0.0,
            )

            (
                next_state,
                reward,
                terminated,
                truncated,
                info,
            ) = env.step(
                action
            )

            episode_reward += (
                reward
            )

            state = next_state
            step += 1

            if (
                terminated
                or truncated
            ):
                break

        episode_rewards.append(
            episode_reward
        )

        episode_lengths.append(
            step
        )

        print(
            f"Episode {episode + 1}:"
        )

        print(
            "  Total Reward:",
            f"{episode_reward:.2f}",
        )

        print(
            "  Episode Length:",
            step,
        )

        reason = (
            "Terminated (fell)"
            if terminated
            else "Truncated (time limit)"
        )

        print(
            "  Reason:",
            reason,
        )

        if (
            "reward_components"
            in info
        ):

            print(
                "  Final Forward Velocity:",
                f"{info['reward_components']['forward_velocity']:.3f} m/s",
            )

        print()

    print("=" * 70)

    print(
        "Average Reward:",
        f"{np.mean(episode_rewards):.2f} "
        f"+/- {np.std(episode_rewards):.2f}",
    )

    print(
        "Average Length:",
        f"{np.mean(episode_lengths):.1f} "
        f"+/- {np.std(episode_lengths):.1f}",
    )

    print("=" * 70)

    env.close()

    return (
        episode_rewards,
        episode_lengths,
    )


def test_with_pose_from_image(
    model_path,
    image_path,
    render=True,
):
    """
    Extract the initial pose from an image using OpenPose,
    then evaluate the trained policy from that pose.
    """

    print()
    print("=" * 70)
    print("IMAGE-DEFINED INITIAL POSE")
    print("=" * 70)

    print(
        "Image:",
        image_path,
    )

    result = extract_pose_with_openpose(
        image_path
    )

    theta_init = np.asarray(
        result["pose_vector_np"],
        dtype=np.float32,
    )

    if theta_init.shape != (
        8,
    ):
        raise ValueError(
            "OpenPose bridge returned "
            f"unexpected pose shape {theta_init.shape}"
        )

    print(
        "[OK] People detected:",
        result["people_detected"],
    )

    print(
        "[OK] Selected person:",
        result[
            "selected_person_number"
        ],
    )

    print(
        "[OK] Selection area:",
        result[
            "selected_person"
        ]["area"],
    )

    print()
    print(
        "Actuator order:"
    )

    for index, (
        name,
        angle,
    ) in enumerate(
        zip(
            result["actuator_order"],
            theta_init,
        )
    ):

        print(
            f"  {index}: "
            f"{name:<16} "
            f"{angle:+.4f} rad"
        )

    print()
    print(
        "Pose dimension:",
        theta_init.shape,
    )

    print(
        "Testing trained policy "
        "from image-defined pose..."
    )

    return test_agent(
        model_path=model_path,
        num_episodes=3,
        render=render,
        initial_pose=theta_init,
    )


def compare_poses(
    model_path
):
    """
    Compare the same trained policy across several initial poses.
    """

    poses = {
        "Standing": np.array(
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            dtype=np.float32,
        ),

        "Mild Forward Pose": np.array(
            [
                0.30,
                -0.40,
                0.10,
                -0.30,
                -0.30,
                -0.40,
                0.30,
                -0.40,
            ],
            dtype=np.float32,
        ),

        "Left-biased Pose": np.array(
            [
                -0.20,
                -0.50,
                0.40,
                -0.20,
                -0.50,
                -0.30,
                0.20,
                -0.20,
            ],
            dtype=np.float32,
        ),

        "Right-biased Pose": np.array(
            [
                0.40,
                -0.20,
                -0.20,
                -0.50,
                -0.20,
                -0.20,
                0.50,
                -0.30,
            ],
            dtype=np.float32,
        ),
    }

    print()
    print("=" * 70)
    print(
        "COMPARING INITIAL POSES"
    )
    print("=" * 70)

    results = {}

    for (
        pose_name,
        pose,
    ) in poses.items():

        print()
        print(
            "Testing pose:",
            pose_name,
        )

        print("-" * 70)

        rewards, lengths = test_agent(
            model_path=model_path,
            num_episodes=3,
            render=False,
            initial_pose=pose,
        )

        results[
            pose_name
        ] = {
            "avg_reward": float(
                np.mean(rewards)
            ),
            "avg_length": float(
                np.mean(lengths)
            ),
        }

    print()
    print("=" * 70)
    print(
        "PERFORMANCE BY INITIAL POSE"
    )
    print("=" * 70)

    print(
        f"{'Pose':<24}"
        f"{'Avg Reward':<16}"
        f"{'Avg Length':<16}"
    )

    print("-" * 56)

    for (
        pose_name,
        stats,
    ) in results.items():

        print(
            f"{pose_name:<24}"
            f"{stats['avg_reward']:<16.2f}"
            f"{stats['avg_length']:<16.1f}"
        )

    print("=" * 70)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a trained humanoid "
            "DQN policy."
        )
    )

    parser.add_argument(
        "--model",
        type=str,
        default=(
            "checkpoints/"
            "dqn_final.pth"
        ),
        help=(
            "Path to trained checkpoint."
        ),
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=5,
        help=(
            "Number of test episodes."
        ),
    )

    parser.add_argument(
        "--no-render",
        action="store_true",
        help=(
            "Disable PyBullet rendering."
        ),
    )

    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help=(
            "Image used to define "
            "the initial pose."
        ),
    )

    parser.add_argument(
        "--compare-poses",
        action="store_true",
        help=(
            "Compare policy performance "
            "across several initial poses."
        ),
    )

    args = parser.parse_args()

    if not os.path.exists(
        args.model
    ):

        print(
            "Error: model file not found:",
            args.model,
        )

        print(
            "Train a model before evaluation."
        )

        sys.exit(
            1
        )

    if args.compare_poses:

        compare_poses(
            args.model
        )

    elif args.image:

        test_with_pose_from_image(
            model_path=args.model,
            image_path=args.image,
            render=not args.no_render,
        )

    else:

        test_agent(
            model_path=args.model,
            num_episodes=args.episodes,
            render=not args.no_render,
        )