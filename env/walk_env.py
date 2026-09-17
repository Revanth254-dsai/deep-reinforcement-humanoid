import os
import time
import numpy as np
import pybullet as p
import pybullet_data

import gymnasium as gym
from gymnasium import spaces


class HumanoidWalkEnv(gym.Env):
    """
    Gymnasium environment for humanoid locomotion in PyBullet.

    Features
    --------
    - Loads a custom humanoid URDF.
    - Dynamically discovers controllable joints.
    - Supports image/pose-conditioned environment resets.
    - Uses discrete torque bins for every controllable joint.
    - Observation contains base state + joint positions/velocities.
    - Reward encourages forward motion and survival while penalizing energy.
    """

    metadata = {
        "render_modes": ["human"],
        "render_fps": 240,
    }

    def __init__(
        self,
        render_mode=None,
        num_bins=5,
        max_episode_steps=500,
        time_step=1.0 / 240.0,
    ):
        super().__init__()

        self.render_mode = render_mode
        self.num_bins = int(num_bins)
        self.max_episode_steps = int(max_episode_steps)
        self.time_step = float(time_step)

        if self.num_bins < 2:
            raise ValueError("num_bins must be at least 2")

        # Reward weights
        self.w_velocity = 1.0
        self.w_alive = 0.10
        self.w_energy = 0.001

        # Fall detection
        self.min_base_height = 0.55
        self.max_tilt = 1.20

        # Initial root height
        self.start_position = [0.0, 0.0, 1.0]

        self.step_count = 0
        self.robot_id = None
        self.plane_id = None

        # ------------------------------------------------------------------
        # Connect to PyBullet
        # ------------------------------------------------------------------
        if self.render_mode == "human":
            self.client_id = p.connect(p.GUI)
        else:
            self.client_id = p.connect(p.DIRECT)

        if self.client_id < 0:
            raise RuntimeError("Could not connect to PyBullet")

        self.urdf_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "humanoid.urdf",
        )

        if not os.path.exists(self.urdf_path):
            raise FileNotFoundError(
                f"Humanoid URDF not found: {self.urdf_path}"
            )

        # Build the world once so that joint information is available
        self._build_world()
        self._discover_controllable_joints()
        self._configure_spaces()

    # ======================================================================
    # WORLD SETUP
    # ======================================================================

    def _build_world(self):
        """Reset PyBullet and load the plane and humanoid."""

        p.resetSimulation(physicsClientId=self.client_id)

        p.setAdditionalSearchPath(
            pybullet_data.getDataPath(),
            physicsClientId=self.client_id,
        )

        p.setGravity(
            0.0,
            0.0,
            -9.81,
            physicsClientId=self.client_id,
        )

        p.setTimeStep(
            self.time_step,
            physicsClientId=self.client_id,
        )

        self.plane_id = p.loadURDF(
            "plane.urdf",
            physicsClientId=self.client_id,
        )

        self.robot_id = p.loadURDF(
            self.urdf_path,
            basePosition=self.start_position,
            useFixedBase=False,
            physicsClientId=self.client_id,
        )

        p.changeDynamics(
            self.plane_id,
            -1,
            lateralFriction=1.0,
            physicsClientId=self.client_id,
        )

    # ======================================================================
    # DYNAMIC JOINT DISCOVERY
    # ======================================================================

    def _discover_controllable_joints(self):
        """
        Discover every movable joint directly from the loaded URDF.

        Fixed joints are ignored automatically.
        """

        self.controllable_joint_indices = []
        self.controllable_joints = []
        self.joint_max_forces = []
        self.joint_lower_limits = []
        self.joint_upper_limits = []

        total_joints = p.getNumJoints(
            self.robot_id,
            physicsClientId=self.client_id,
        )

        for joint_index in range(total_joints):
            info = p.getJointInfo(
                self.robot_id,
                joint_index,
                physicsClientId=self.client_id,
            )

            joint_name = info[1].decode("utf-8")
            joint_type = info[2]

            if joint_type == p.JOINT_FIXED:
                continue

            self.controllable_joint_indices.append(joint_index)
            self.controllable_joints.append(joint_name)

            lower_limit = float(info[8])
            upper_limit = float(info[9])
            max_force = float(info[10])

            if max_force <= 0:
                max_force = 50.0

            self.joint_lower_limits.append(lower_limit)
            self.joint_upper_limits.append(upper_limit)
            self.joint_max_forces.append(max_force)

        self.num_joints = len(self.controllable_joint_indices)

        if self.num_joints == 0:
            raise RuntimeError(
                "No controllable joints were discovered in humanoid.urdf"
            )

        self.joint_max_forces = np.asarray(
            self.joint_max_forces,
            dtype=np.float32,
        )

        self.joint_lower_limits = np.asarray(
            self.joint_lower_limits,
            dtype=np.float32,
        )

        self.joint_upper_limits = np.asarray(
            self.joint_upper_limits,
            dtype=np.float32,
        )

        # Disable PyBullet's default velocity motors.
        # This is required before using TORQUE_CONTROL.
        for joint_index in self.controllable_joint_indices:
            p.setJointMotorControl2(
                bodyUniqueId=self.robot_id,
                jointIndex=joint_index,
                controlMode=p.VELOCITY_CONTROL,
                force=0.0,
                physicsClientId=self.client_id,
            )

    # ======================================================================
    # GYMNASIUM SPACES
    # ======================================================================

    def _configure_spaces(self):
        """
        Observation:

        Base:
            position             3
            quaternion           4
            linear velocity      3
            angular velocity     3

        Joints:
            positions            num_joints
            velocities           num_joints

        With 8 controllable joints:
            3 + 4 + 3 + 3 + 8 + 8 = 29 dimensions
        """

        state_dim = 13 + (2 * self.num_joints)

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(state_dim,),
            dtype=np.float32,
        )

        # Every joint independently chooses one of num_bins torque values.
        self.action_space = spaces.MultiDiscrete(
            np.full(
                self.num_joints,
                self.num_bins,
                dtype=np.int64,
            )
        )

    # ======================================================================
    # OBSERVATION
    # ======================================================================

    def _get_observation(self):
        base_position, base_orientation = p.getBasePositionAndOrientation(
            self.robot_id,
            physicsClientId=self.client_id,
        )

        base_linear_velocity, base_angular_velocity = p.getBaseVelocity(
            self.robot_id,
            physicsClientId=self.client_id,
        )

        joint_states = p.getJointStates(
            self.robot_id,
            self.controllable_joint_indices,
            physicsClientId=self.client_id,
        )

        joint_positions = [state[0] for state in joint_states]
        joint_velocities = [state[1] for state in joint_states]

        observation = np.concatenate(
            [
                np.asarray(base_position, dtype=np.float32),
                np.asarray(base_orientation, dtype=np.float32),
                np.asarray(base_linear_velocity, dtype=np.float32),
                np.asarray(base_angular_velocity, dtype=np.float32),
                np.asarray(joint_positions, dtype=np.float32),
                np.asarray(joint_velocities, dtype=np.float32),
            ]
        )

        return observation.astype(np.float32)

    # ======================================================================
    # RESET
    # ======================================================================

    def reset(self, seed=None, options=None, initial_pose=None):
        """
        Reset the physics world.

        Parameters
        ----------
        initial_pose:
            Optional array of joint angles.

            If longer than the number of controllable joints it is truncated.
            If shorter it is zero-padded.
        """

        super().reset(seed=seed)

        self.step_count = 0

        self._build_world()
        self._discover_controllable_joints()

        if initial_pose is None:
            pose = np.zeros(self.num_joints, dtype=np.float32)
        else:
            pose = np.asarray(
                initial_pose,
                dtype=np.float32,
            ).reshape(-1)

            if pose.size < self.num_joints:
                pose = np.pad(
                    pose,
                    (0, self.num_joints - pose.size),
                    mode="constant",
                )

            elif pose.size > self.num_joints:
                pose = pose[: self.num_joints]

        # Respect URDF limits.
        pose = np.clip(
            pose,
            self.joint_lower_limits,
            self.joint_upper_limits,
        )

        for i, joint_index in enumerate(
            self.controllable_joint_indices
        ):
            p.resetJointState(
                bodyUniqueId=self.robot_id,
                jointIndex=joint_index,
                targetValue=float(pose[i]),
                targetVelocity=0.0,
                physicsClientId=self.client_id,
            )

            # resetJointState can reactivate default motors,
            # so explicitly disable them again.
            p.setJointMotorControl2(
                bodyUniqueId=self.robot_id,
                jointIndex=joint_index,
                controlMode=p.VELOCITY_CONTROL,
                force=0.0,
                physicsClientId=self.client_id,
            )

        observation = self._get_observation()

        info = {
            "initial_pose": pose.copy(),
            "num_joints": self.num_joints,
        }

        return observation, info

    # ======================================================================
    # ACTION → TORQUE
    # ======================================================================

    def _action_to_torque(self, action):
        """
        Convert discrete action bins into continuous joint torques.

        Example with five bins:

            0 -> -100 %
            1 ->  -50 %
            2 ->    0 %
            3 ->  +50 %
            4 -> +100 %

        Each value is scaled by that joint's URDF torque limit.
        """

        action = np.asarray(
            action,
            dtype=np.int64,
        ).reshape(-1)

        if action.size != self.num_joints:
            raise ValueError(
                f"Expected {self.num_joints} joint actions, "
                f"received {action.size}"
            )

        action = np.clip(
            action,
            0,
            self.num_bins - 1,
        )

        normalized = (
            (2.0 * action.astype(np.float32))
            / float(self.num_bins - 1)
        ) - 1.0

        torques = normalized * self.joint_max_forces

        return torques.astype(np.float32)

    # ======================================================================
    # REWARD
    # ======================================================================

    def _calculate_reward(self, torques):
        base_position, base_orientation = p.getBasePositionAndOrientation(
            self.robot_id,
            physicsClientId=self.client_id,
        )

        linear_velocity, _ = p.getBaseVelocity(
            self.robot_id,
            physicsClientId=self.client_id,
        )

        joint_states = p.getJointStates(
            self.robot_id,
            self.controllable_joint_indices,
            physicsClientId=self.client_id,
        )

        joint_velocities = np.asarray(
            [state[1] for state in joint_states],
            dtype=np.float32,
        )

        # Positive X velocity represents forward movement.
        forward_velocity = float(linear_velocity[0])

        # Alive reward encourages remaining upright.
        alive_bonus = 1.0

        # Mechanical effort approximation.
        energy_penalty = float(
            np.mean(
                np.abs(
                    torques * joint_velocities
                )
            )
        )

        reward = (
            self.w_velocity * forward_velocity
            + self.w_alive * alive_bonus
            - self.w_energy * energy_penalty
        )

        roll, pitch, _ = p.getEulerFromQuaternion(
            base_orientation
        )

        reward_components = {
            "forward_velocity": forward_velocity,
            "alive_bonus": alive_bonus,
            "energy_penalty": energy_penalty,
            "base_height": float(base_position[2]),
            "roll": float(roll),
            "pitch": float(pitch),
        }

        return float(reward), reward_components

    # ======================================================================
    # TERMINATION
    # ======================================================================

    def _has_fallen(self):
        base_position, base_orientation = p.getBasePositionAndOrientation(
            self.robot_id,
            physicsClientId=self.client_id,
        )

        roll, pitch, _ = p.getEulerFromQuaternion(
            base_orientation
        )

        too_low = base_position[2] < self.min_base_height

        too_tilted = (
            abs(roll) > self.max_tilt
            or abs(pitch) > self.max_tilt
        )

        return bool(too_low or too_tilted)

    # ======================================================================
    # STEP
    # ======================================================================

    def step(self, action):
        torques = self._action_to_torque(action)

        for i, joint_index in enumerate(
            self.controllable_joint_indices
        ):
            p.setJointMotorControl2(
                bodyUniqueId=self.robot_id,
                jointIndex=joint_index,
                controlMode=p.TORQUE_CONTROL,
                force=float(torques[i]),
                physicsClientId=self.client_id,
            )

        p.stepSimulation(
            physicsClientId=self.client_id
        )

        self.step_count += 1

        reward, reward_components = self._calculate_reward(
            torques
        )

        terminated = self._has_fallen()

        truncated = (
            self.step_count >= self.max_episode_steps
        )

        observation = self._get_observation()

        info = {
            "reward_components": reward_components,
            "torques": torques.copy(),
            "step": self.step_count,
        }

        if self.render_mode == "human":
            time.sleep(self.time_step)

        return (
            observation,
            reward,
            terminated,
            truncated,
            info,
        )

    # ======================================================================
    # RENDER / CLOSE
    # ======================================================================

    def render(self):
        # PyBullet GUI rendering is handled directly by p.GUI.
        return None

    def close(self):
        if (
            hasattr(self, "client_id")
            and self.client_id >= 0
            and p.isConnected(self.client_id)
        ):
            p.disconnect(
                physicsClientId=self.client_id
            )

            self.client_id = -1