"""
Complete validation for the image-to-humanoid RL pipeline.

Tests:

1. PyBullet humanoid URDF
2. Dynamic actuator discovery
3. Gymnasium reset/step API
4. Torque-based control
5. Reward computation
6. Python 3.10 -> Python 3.7 OpenPose bridge
7. BODY_25 primary-person selection
8. Image-derived 8-joint pose
9. Pose-conditioned PyBullet reset
10. Short physics rollout
"""

import argparse
import os
import sys

import numpy as np
import torch


PROJECT_ROOT = os.path.dirname(
    os.path.abspath(__file__)
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(
        0,
        PROJECT_ROOT,
    )


from env.walk_env import HumanoidWalkEnv
from pose.openpose_bridge import extract_pose_with_openpose
from agent.network import DQNetwork, DuelingDQNetwork


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Complete humanoid environment "
            "and image-pose integration test."
        )
    )

    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=(
            "Optional trained model checkpoint."
        ),
    )

    parser.add_argument(
        "--duration",
        type=int,
        default=120,
        help=(
            "Walking-demo duration in seconds."
        ),
    )

    parser.add_argument(
        "--no-time-limit",
        action="store_true",
        help=(
            "Disable episode time limit "
            "during trained-model demo."
        ),
    )

    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Disable PyBullet GUI.",
    )

    args = parser.parse_args()

    print()
    print("=" * 70)
    print(
        "HUMANOID RL PIPELINE - COMPLETE TEST"
    )
    print("=" * 70)

    # =====================================================================
    # MODULE 2
    # PyBullet Environment
    # =====================================================================

    print()
    print("[1] HUMANOID ASSET")
    print("-" * 70)

    urdf_path = os.path.join(
        PROJECT_ROOT,
        "env",
        "humanoid.urdf",
    )

    if not os.path.exists(
        urdf_path
    ):
        raise FileNotFoundError(
            f"URDF missing: {urdf_path}"
        )

    print(
        "[OK] URDF:",
        urdf_path,
    )

    render_mode = (
        None
        if args.no_render
        else "human"
    )

    env = HumanoidWalkEnv(
        render_mode=render_mode
    )

    print(
        "[OK] PyBullet initialized"
    )

    print(
        "[OK] Controllable joints:",
        env.num_joints,
    )

    print(
        "[OK] Joint names:",
        env.controllable_joints,
    )

    print(
        "[OK] Observation space:",
        env.observation_space,
    )

    print(
        "[OK] Action space:",
        env.action_space,
    )

    assert env.num_joints == 8

    assert env.observation_space.shape == (
        29,
    )

    # =====================================================================
    # RESET TEST
    # =====================================================================

    print()
    print("[2] RESET API")
    print("-" * 70)

    obs, info = env.reset()

    assert obs.shape == (
        29,
    )

    assert np.all(
        np.isfinite(obs)
    )

    print(
        "[OK] Default reset"
    )

    print(
        "[OK] Observation shape:",
        obs.shape,
    )

    custom_pose = np.array(
        [
            0.15,
            -0.30,
            -0.10,
            -0.20,
            -0.25,
            -0.50,
            0.20,
            -0.40,
        ],
        dtype=np.float32,
    )

    obs, info = env.reset(
        initial_pose=custom_pose
    )

    applied_pose = np.asarray(
        info["initial_pose"],
        dtype=np.float32,
    )

    assert applied_pose.shape == (
        8,
    )

    assert np.allclose(
        custom_pose,
        applied_pose,
        atol=1e-5,
    )

    print(
        "[OK] Custom pose reset"
    )

    print(
        "[OK] Pose applied without clipping"
    )

    # =====================================================================
    # STEP / PHYSICS TEST
    # =====================================================================

    print()
    print("[3] PHYSICS + TORQUE CONTROL")
    print("-" * 70)

    completed_steps = 0

    for step in range(
        100
    ):

        action = (
            env.action_space.sample()
        )

        (
            obs,
            reward,
            terminated,
            truncated,
            info,
        ) = env.step(
            action
        )

        completed_steps += 1

        assert obs.shape == (
            29,
        )

        assert np.all(
            np.isfinite(obs)
        )

        assert np.isfinite(
            reward
        )

        if step % 25 == 0:

            components = info[
                "reward_components"
            ]

            print(
                f"Step {step:3d} | "
                f"Reward={reward:+.4f} | "
                f"Velocity="
                f"{components['forward_velocity']:+.4f} | "
                f"Height="
                f"{components['base_height']:.4f}"
            )

        if (
            terminated
            or truncated
        ):

            print(
                "Episode ended naturally at "
                f"step {completed_steps}"
            )

            break

    print(
        "[OK] Physics rollout completed"
    )

    print(
        "[OK] Steps:",
        completed_steps,
    )

    # =====================================================================
    # FULL MODULE 1 -> MODULE 2 INTEGRATION
    # =====================================================================

    print()
    print("=" * 70)
    print(
        "[4] FULL IMAGE -> OPENPOSE -> HUMANOID TEST"
    )
    print("=" * 70)

    image_path = os.path.join(
        PROJECT_ROOT,
        "data",
        "input_images",
        "sample2.jpeg",
    )

    if not os.path.exists(
        image_path
    ):
        env.close()

        raise FileNotFoundError(
            f"Integration image missing: {image_path}"
        )

    print(
        "Image:",
        image_path,
    )

    # Python 3.10 -> Python 3.7 -> OpenPose
    pose_result = extract_pose_with_openpose(
        image_path
    )

    theta_init = pose_result[
        "pose_vector_np"
    ]

    print(
        "[OK] OpenPose BODY_25 completed"
    )

    print(
        "[OK] People detected:",
        pose_result["people_detected"],
    )

    print(
        "[OK] Selected person:",
        pose_result[
            "selected_person_number"
        ],
    )

    print(
        "[OK] Bounding-box area:",
        pose_result[
            "selected_person"
        ]["area"],
    )

    print(
        "[OK] Pose-vector shape:",
        theta_init.shape,
    )

    print(
        "[OK] Pose:",
        theta_init,
    )

    assert theta_init.shape == (
        8,
    )

    assert np.all(
        np.isfinite(theta_init)
    )

    # ---------------------------------------------------------------------
    # Confirm ordering between BODY_25 conversion and PyBullet URDF
    # ---------------------------------------------------------------------

    pose_order = list(
        pose_result[
            "actuator_order"
        ]
    )

    environment_order = list(
        env.controllable_joints
    )

    print()
    print(
        "Pose order:",
        pose_order,
    )

    print(
        "Environment order:",
        environment_order,
    )

    assert (
        pose_order
        == environment_order
    ), (
        "Pose actuator order does not "
        "match PyBullet joint order."
    )

    print(
        "[OK] Actuator order matches"
    )

    # ---------------------------------------------------------------------
    # Reset simulation using actual image pose
    # ---------------------------------------------------------------------

    obs, reset_info = env.reset(
        initial_pose=theta_init
    )

    applied_pose = np.asarray(
        reset_info[
            "initial_pose"
        ],
        dtype=np.float32,
    )

    print()
    print(
        "Requested image pose:",
        theta_init,
    )

    print(
        "Applied simulator pose:",
        applied_pose,
    )

    assert np.allclose(
        theta_init,
        applied_pose,
        atol=1e-5,
    ), (
        "Image pose was clipped or changed "
        "by PyBullet joint limits."
    )

    print(
        "[OK] Image-derived pose applied "
        "without joint-limit clipping"
    )

    # ---------------------------------------------------------------------
    # Short rollout from image-defined pose
    # ---------------------------------------------------------------------

    print()
    print(
        "Running physics from "
        "image-defined pose..."
    )

    image_pose_steps = 0

    for step in range(
        200
    ):

        action = (
            env.action_space.sample()
        )

        (
            obs,
            reward,
            terminated,
            truncated,
            info,
        ) = env.step(
            action
        )

        image_pose_steps += 1

        assert np.all(
            np.isfinite(obs)
        )

        assert np.isfinite(
            reward
        )

        if step % 50 == 0:

            velocity = info[
                "reward_components"
            ][
                "forward_velocity"
            ]

            print(
                f"Step {step:3d} | "
                f"Reward={reward:+.4f} | "
                f"Velocity={velocity:+.4f}"
            )

        if (
            terminated
            or truncated
        ):

            print(
                "Image-pose episode ended "
                f"at step {image_pose_steps}"
            )

            break

    print()
    print(
        "[OK] FULL PIPELINE PASSED:"
    )

    print(
        "     Image"
    )

    print(
        "       -> OpenPose BODY_25"
    )

    print(
        "       -> Primary-person selection"
    )

    print(
        "       -> 8-joint pose conversion"
    )

    print(
        "       -> PyBullet pose reset"
    )

    print(
        "       -> Torque-controlled simulation"
    )

    # =====================================================================
    # OPTIONAL TRAINED MODEL
    # =====================================================================

    if (
        args.model is None
        and os.path.exists(
            os.path.join(
                "checkpoints",
                "dqn_final.pth",
            )
        )
    ):

        args.model = os.path.join(
            "checkpoints",
            "dqn_final.pth",
        )

    if (
        args.model is not None
        and os.path.exists(
            args.model
        )
    ):

        print()
        print("=" * 70)
        print(
            "[5] TRAINED MODEL LOAD TEST"
        )
        print("=" * 70)

        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        checkpoint = torch.load(
            args.model,
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
                checkpoint[
                    "policy_net"
                ]
            )

            print(
                "[OK] Loaded Dueling DQN"
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
                checkpoint[
                    "policy_net"
                ]
            )

            print(
                "[OK] Loaded standard DQN"
            )

        model.eval()

    env.close()

    # =====================================================================
    # FINAL RESULT
    # =====================================================================

    print()
    print("=" * 70)
    print("ALL CORE TESTS PASSED")
    print("=" * 70)

    print(
        "[OK] Custom humanoid URDF"
    )

    print(
        "[OK] Dynamic joint discovery"
    )

    print(
        "[OK] Gymnasium API"
    )

    print(
        "[OK] Torque-based control"
    )

    print(
        "[OK] OpenPose BODY_25"
    )

    print(
        "[OK] Multi-person selection"
    )

    print(
        "[OK] 8-joint pose conversion"
    )

    print(
        "[OK] Image-conditioned reset"
    )

    print(
        "[OK] PyBullet simulation"
    )


if __name__ == "__main__":
    main()