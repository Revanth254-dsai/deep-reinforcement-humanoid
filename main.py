"""
ISL 2025 Lab Project
Deep Reinforcement Learning for Humanoid Locomotion
from an Image-Defined Initial Pose

Pipeline:
    Image
      -> OpenPose BODY_25
      -> Primary-person selection
      -> 8-joint pose vector
      -> PyBullet humanoid reset
      -> Dueling DQN training
"""

import os
import sys
import numpy as np

PROJECT_ROOT = os.path.dirname(
    os.path.abspath(__file__)
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


from pose.openpose_bridge import extract_pose_with_openpose
from env.walk_env import HumanoidWalkEnv
from agent.train_dqn import DQNAgent, train


def main():

    print("=" * 70)
    print("ISL 2025 Lab Project")
    print(
        "Deep RL for Humanoid Locomotion "
        "from Image-Defined Initial Pose"
    )
    print("=" * 70)

    # =====================================================================
    # MODULE 1
    # Pose Estimation and Initial Pose Extraction
    # =====================================================================

    print()
    print("[MODULE 1] Pose Estimation & Initial Pose Extraction")
    print("-" * 70)

    image_path = os.path.join(
        PROJECT_ROOT,
        "data",
        "input_images",
        "sample2.jpeg",
    )

    print(
        "Processing image:",
        image_path,
    )

    # Python 3.10 calls the Python 3.7 OpenPose worker.
    pose_result = extract_pose_with_openpose(
        image_path
    )

    theta_init = pose_result[
        "pose_vector_np"
    ]

    print(
        "[OK] OpenPose BODY_25 extraction completed"
    )

    print(
        "[OK] People detected:",
        pose_result["people_detected"],
    )

    print(
        "[OK] Selected primary person:",
        pose_result["selected_person_number"],
    )

    print(
        "[OK] Selection method: "
        "largest bounding-box area"
    )

    print(
        "[OK] Selected bounding-box area:",
        pose_result[
            "selected_person"
        ]["area"],
    )

    print()
    print(
        "[OK] Initial 8-joint pose vector:"
    )

    for index, (
        joint_name,
        angle
    ) in enumerate(
        zip(
            pose_result["actuator_order"],
            theta_init,
        )
    ):

        print(
            f"  {index}: "
            f"{joint_name:<16} "
            f"{angle:+.4f} rad "
            f"({np.degrees(angle):+.2f} deg)"
        )

    # =====================================================================
    # MODULE 2
    # Simulation / Physics Environment
    # =====================================================================

    print()
    print("[MODULE 2] Simulation (Physics Environment)")
    print("-" * 70)

    env = HumanoidWalkEnv(
        render_mode=None
    )

    print(
        "[OK] Gymnasium-compatible environment created"
    )

    print(
        "[OK] Observation space:",
        env.observation_space.shape,
    )

    print(
        "[OK] Action space:",
        env.action_space,
    )

    print(
        "[OK] Controllable joints:",
        env.num_joints,
    )

    print(
        "[OK] Discrete torque bins per joint:",
        env.num_bins,
    )

    print()
    print(
        "Environment actuator order:"
    )

    for index, joint_name in enumerate(
        env.controllable_joints
    ):
        print(
            f"  {index}: {joint_name}"
        )

    # ---------------------------------------------------------------------
    # Verify that OpenPose actuator ordering matches the simulator.
    # ---------------------------------------------------------------------

    expected_order = list(
        pose_result["actuator_order"]
    )

    actual_order = list(
        env.controllable_joints
    )

    if expected_order != actual_order:
        env.close()

        raise RuntimeError(
            "Pose-vector actuator order does not "
            "match the PyBullet humanoid.\n"
            f"Pose order: {expected_order}\n"
            f"Environment order: {actual_order}"
        )

    print(
        "[OK] OpenPose pose-vector order matches "
        "PyBullet actuator order"
    )

    # ---------------------------------------------------------------------
    # Image-conditioned environment reset
    # ---------------------------------------------------------------------

    obs, reset_info = env.reset(
        initial_pose=theta_init
    )

    applied_pose = np.asarray(
        reset_info["initial_pose"],
        dtype=np.float32,
    )

    print()
    print(
        "[OK] Humanoid reset using image-derived pose"
    )

    print(
        "[OK] Observation shape:",
        obs.shape,
    )

    print(
        "[OK] Applied pose:",
        applied_pose,
    )

    # The converted image angles should already be inside
    # the URDF limits. Verify that reset did not alter them.
    if not np.allclose(
        theta_init,
        applied_pose,
        atol=1e-5,
    ):
        env.close()

        raise RuntimeError(
            "Image-derived pose was clipped by "
            "the simulator joint limits.\n"
            f"Requested: {theta_init}\n"
            f"Applied:   {applied_pose}"
        )

    print(
        "[OK] Image-derived pose required no "
        "joint-limit clipping"
    )

    # =====================================================================
    # MODULE 3
    # DQN Agent
    # =====================================================================

    print()
    print("[MODULE 3] Control (DQN Agent)")
    print("-" * 70)

    agent = DQNAgent(
        env=env,
        use_dueling=True,
        learning_rate=1e-4,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.985,
        buffer_capacity=30000,
        batch_size=32,
    )

    print(
        "[OK] Dueling DQN agent created"
    )

    print(
        "[OK] Experience replay enabled"
    )

    print(
        "[OK] Epsilon-greedy exploration configured"
    )

    print(
        "[OK] Action discretization:",
        f"{env.num_joints} joints x "
        f"{env.num_bins} bins",
    )

    print(
        "[OK] Reward components: "
        "forward velocity + alive bonus - energy penalty"
    )

    # =====================================================================
    # TRAINING
    # =====================================================================

    print()
    print("[TRAINING] DQN Training with Pose Library")
    print("-" * 70)

    # Include the image-derived pose together with other valid
    # starting configurations so episodes can generalize across poses.
    pose_library = [
        theta_init.copy(),

        np.zeros(
            env.num_joints,
            dtype=np.float32,
        ),

        np.array(
            [
                0.3,
                -0.5,
                -0.2,
                -0.3,
                -0.5,
                -0.2,
                0.1,
                -0.1,
            ],
            dtype=np.float32,
        ),

        np.array(
            [
                -0.2,
                -0.8,
                -0.3,
                -0.2,
                -0.8,
                -0.3,
                -0.1,
                -0.1,
            ],
            dtype=np.float32,
        ),
    ]

    print(
        "[OK] Pose library created with",
        len(pose_library),
        "initial poses",
    )

    print(
        "[OK] Image-derived pose included in "
        "training pose library"
    )

    print()
    print(
        "Starting training "
        "(150 episodes)..."
    )

    rewards, lengths, losses = train(
        env=env,
        agent=agent,
        num_episodes=150,
        max_steps_per_episode=500,
        save_freq=30,
        log_freq=10,
        save_dir="checkpoints",
        pose_library=pose_library,
    )

    # =====================================================================
    # SAVE MODEL
    # =====================================================================

    os.makedirs(
        "checkpoints",
        exist_ok=True,
    )

    final_model_path = os.path.join(
        "checkpoints",
        "dqn_final.pth",
    )

    agent.save(
        final_model_path
    )

    # =====================================================================
    # SUMMARY
    # =====================================================================

    print()
    print("=" * 70)
    print("PROJECT OUTPUTS")
    print("=" * 70)

    print()
    print(
        "1. OpenPose BODY_25:"
    )
    print(
        "   People detected:",
        pose_result["people_detected"],
    )
    print(
        "   Selected person:",
        pose_result["selected_person_number"],
    )

    print()
    print(
        "2. Image-to-Pose Conversion:"
    )
    print(
        "   Pose dimension:",
        theta_init.shape,
    )
    print(
        "   Actuator order:",
        pose_result["actuator_order"],
    )

    print()
    print(
        "3. Gymnasium / PyBullet Environment:"
    )
    print(
        "   Observation dimensions:",
        env.observation_space.shape[0],
    )
    print(
        "   Controllable joints:",
        env.num_joints,
    )
    print(
        "   Torque bins:",
        env.num_bins,
    )

    print()
    print(
        "4. DQN:"
    )
    print(
        "   Architecture: Dueling DQN"
    )
    print(
        "   Experience replay: enabled"
    )
    print(
        "   Exploration: epsilon-greedy"
    )

    print()
    print(
        "5. Reward:"
    )
    print(
        "   Forward velocity + alive bonus "
        "- energy expenditure"
    )

    print()
    print(
        "6. Trained Model:"
    )
    print(
        "   Saved to:",
        final_model_path,
    )

    print()
    print("=" * 70)
    print("TRAINING SUMMARY")
    print("=" * 70)

    print(
        "Total episodes:",
        len(rewards),
    )

    if rewards:
        print(
            "Average reward (last 10):",
            f"{np.mean(rewards[-10:]):.2f}",
        )

    if lengths:
        print(
            "Average episode length (last 10):",
            f"{np.mean(lengths[-10:]):.1f}",
        )

    if losses:
        print(
            "Final recorded loss:",
            f"{losses[-1]:.6f}",
        )

    env.close()

    print()
    print("=" * 70)
    print("PROJECT COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()