import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.optim as optim


# -------------------------------------------------------------------------
# Project imports
# -------------------------------------------------------------------------

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


from agent.network import DQNetwork, DuelingDQNetwork
from agent.replay import ReplayBuffer, PrioritizedReplayBuffer
from env.walk_env import HumanoidWalkEnv


# =========================================================================
# Pose dataset utilities
# =========================================================================

def _load_pose_vectors_from_dir(
    poses_dir: str,
    target_dim: int = 8,
):
    """
    Load pose vectors stored as .npy files.

    Parameters
    ----------
    poses_dir : str
        Directory containing pose-vector files.

    target_dim : int
        Desired pose dimension.

    Returns
    -------
    list[np.ndarray]
        Pose vectors normalized to target_dim.
    """

    poses = []

    if (
        poses_dir is None
        or not os.path.exists(poses_dir)
    ):
        return poses

    for root, _, files in os.walk(
        poses_dir
    ):

        for filename in sorted(files):

            if not filename.lower().endswith(
                ".npy"
            ):
                continue

            file_path = os.path.join(
                root,
                filename,
            )

            try:

                vector = np.load(
                    file_path
                )

                vector = np.asarray(
                    vector,
                    dtype=np.float32,
                ).reshape(
                    -1
                )

                if vector.shape[0] < target_dim:

                    vector = np.pad(
                        vector,
                        (
                            0,
                            target_dim
                            - vector.shape[0],
                        ),
                        mode="constant",
                    )

                elif vector.shape[0] > target_dim:

                    vector = vector[
                        :target_dim
                    ]

                # General safety bound.
                vector = np.clip(
                    vector,
                    -np.pi,
                    np.pi,
                ).astype(
                    np.float32
                )

                if np.all(
                    np.isfinite(vector)
                ):

                    poses.append(
                        vector
                    )

            except Exception as exc:

                print(
                    f"[WARNING] Could not load "
                    f"{file_path}: {exc}"
                )

    return poses


def _prefill_replay_with_rollouts(
    env,
    agent,
    poses,
    num_rollouts: int = 50,
    steps_per_rollout: int = 50,
):
    """
    Prefill the replay buffer using short exploratory rollouts
    from pose-library initial states.
    """

    if not poses:
        return 0

    original_epsilon = (
        agent.epsilon
    )

    # Keep exploration reasonably high during warm-start rollout.
    agent.epsilon = max(
        0.5,
        original_epsilon,
    )

    added = 0

    try:

        for rollout_index in range(
            num_rollouts
        ):

            pose = poses[
                rollout_index
                % len(poses)
            ]

            state, _ = env.reset(
                initial_pose=pose
            )

            for _ in range(
                steps_per_rollout
            ):

                action = agent.select_action(
                    state
                )

                (
                    next_state,
                    reward,
                    terminated,
                    truncated,
                    _,
                ) = env.step(
                    action
                )

                done = (
                    terminated
                    or truncated
                )

                agent.store_transition(
                    state,
                    action,
                    reward,
                    next_state,
                    done,
                )

                added += 1

                state = next_state

                if done:
                    break

    finally:

        agent.epsilon = (
            original_epsilon
        )

    return added


# =========================================================================
# DQN Agent
# =========================================================================

class DQNAgent:
    """
    DQN agent for humanoid locomotion.

    Supports:
    - Standard DQN
    - Dueling DQN
    - Target network
    - Soft or hard target updates
    - Experience replay
    - Prioritized experience replay
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
        device=None,
    ):

        self.env = env

        self.state_dim = (
            env.observation_space.shape[
                0
            ]
        )

        self.num_joints = (
            env.num_joints
        )

        self.num_bins = (
            env.num_bins
        )

        self.gamma = gamma

        self.batch_size = (
            batch_size
        )

        self.epsilon = (
            epsilon_start
        )

        self.epsilon_end = (
            epsilon_end
        )

        self.epsilon_decay = (
            epsilon_decay
        )

        self.target_update_freq = (
            target_update_freq
        )

        self.soft_update_tau = (
            soft_update_tau
        )

        self.use_soft_update = (
            use_soft_update
        )

        self.use_prioritized = (
            use_prioritized
        )

        # -----------------------------------------------------------------
        # Device
        # -----------------------------------------------------------------

        if device is not None:

            self.device = (
                device
            )

        else:

            self.device = torch.device(
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )

        print(
            f"Using device: {self.device}"
        )

        # -----------------------------------------------------------------
        # Networks
        # -----------------------------------------------------------------

        network_class = (
            DuelingDQNetwork
            if use_dueling
            else DQNetwork
        )

        self.policy_net = network_class(
            self.state_dim,
            self.num_joints,
            self.num_bins,
        ).to(
            self.device
        )

        self.target_net = network_class(
            self.state_dim,
            self.num_joints,
            self.num_bins,
        ).to(
            self.device
        )

        self.target_net.load_state_dict(
            self.policy_net.state_dict()
        )

        self.target_net.eval()

        # -----------------------------------------------------------------
        # Optimizer
        # -----------------------------------------------------------------

        self.optimizer = optim.Adam(
            self.policy_net.parameters(),
            lr=learning_rate,
        )

        # -----------------------------------------------------------------
        # Replay buffer
        # -----------------------------------------------------------------

        if use_prioritized:

            self.replay_buffer = (
                PrioritizedReplayBuffer(
                    capacity=buffer_capacity
                )
            )

        else:

            self.replay_buffer = (
                ReplayBuffer(
                    capacity=buffer_capacity
                )
            )

        # -----------------------------------------------------------------
        # Training counters
        # -----------------------------------------------------------------

        self.steps_done = 0
        self.episodes_done = 0


    def select_action(
        self,
        state,
    ):
        """
        Select a joint-action vector using epsilon-greedy exploration.
        """

        return self.policy_net.get_action(
            state,
            epsilon=self.epsilon,
        )


    def store_transition(
        self,
        state,
        action,
        reward,
        next_state,
        done,
    ):
        """
        Add one transition to replay memory.
        """

        self.replay_buffer.push(
            state,
            action,
            reward,
            next_state,
            done,
        )


    def update_epsilon(
        self,
    ):
        """
        Decay exploration rate.
        """

        self.epsilon = max(
            self.epsilon_end,
            self.epsilon
            * self.epsilon_decay,
        )


    def soft_update_target(
        self,
    ):
        """
        Soft target-network update.
        """

        for (
            target_parameter,
            policy_parameter,
        ) in zip(
            self.target_net.parameters(),
            self.policy_net.parameters(),
        ):

            target_parameter.data.copy_(
                self.soft_update_tau
                * policy_parameter.data
                + (
                    1.0
                    - self.soft_update_tau
                )
                * target_parameter.data
            )


    def hard_update_target(
        self,
    ):
        """
        Copy policy-network parameters directly to the target network.
        """

        self.target_net.load_state_dict(
            self.policy_net.state_dict()
        )


    def train_step(
        self,
    ):
        """
        Perform one DQN optimization step.

        Returns
        -------
        float or None
            Loss value, or None if replay memory does not yet
            contain enough samples.
        """

        if (
            len(self.replay_buffer)
            < self.batch_size
        ):

            return None

        # -----------------------------------------------------------------
        # Sample replay memory
        # -----------------------------------------------------------------

        if self.use_prioritized:

            (
                states,
                actions,
                rewards,
                next_states,
                dones,
                indices,
                weights,
            ) = self.replay_buffer.sample(
                self.batch_size,
                beta=0.4,
            )

            weights = torch.as_tensor(
                weights,
                dtype=torch.float32,
                device=self.device,
            )

        else:

            (
                states,
                actions,
                rewards,
                next_states,
                dones,
            ) = self.replay_buffer.sample(
                self.batch_size
            )

            weights = torch.ones(
                self.batch_size,
                dtype=torch.float32,
                device=self.device,
            )

        # -----------------------------------------------------------------
        # Convert batch to tensors
        # -----------------------------------------------------------------

        states = torch.as_tensor(
            states,
            dtype=torch.float32,
            device=self.device,
        )

        actions = torch.as_tensor(
            actions,
            dtype=torch.long,
            device=self.device,
        )

        rewards = torch.as_tensor(
            rewards,
            dtype=torch.float32,
            device=self.device,
        )

        next_states = torch.as_tensor(
            next_states,
            dtype=torch.float32,
            device=self.device,
        )

        dones = torch.as_tensor(
            dones,
            dtype=torch.float32,
            device=self.device,
        )

        # -----------------------------------------------------------------
        # Current Q-values
        # -----------------------------------------------------------------

        q_values = self.policy_net(
            states
        )

        # q_values:
        # (batch_size, num_joints, num_bins)

        q_values_selected = torch.gather(
            q_values,
            dim=2,
            index=actions.unsqueeze(
                2
            ),
        ).squeeze(
            2
        )

        # Aggregate across independently discretized joints.
        q_values_selected = (
            q_values_selected.mean(
                dim=1
            )
        )

        # -----------------------------------------------------------------
        # Bellman target
        # -----------------------------------------------------------------

        with torch.no_grad():

            next_q_values = (
                self.target_net(
                    next_states
                )
            )

            next_q_values_max = (
                next_q_values.max(
                    dim=2
                )[0].mean(
                    dim=1
                )
            )

            target_q_values = (
                rewards
                + (
                    1.0
                    - dones
                )
                * self.gamma
                * next_q_values_max
            )

        # -----------------------------------------------------------------
        # TD error / loss
        # -----------------------------------------------------------------

        td_errors = (
            target_q_values
            - q_values_selected
        )

        loss = (
            weights
            * td_errors.pow(
                2
            )
        ).mean()

        # -----------------------------------------------------------------
        # Optimization
        # -----------------------------------------------------------------

        self.optimizer.zero_grad()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            self.policy_net.parameters(),
            max_norm=10.0,
        )

        self.optimizer.step()

        # -----------------------------------------------------------------
        # Prioritized replay priorities
        # -----------------------------------------------------------------

        if self.use_prioritized:

            priorities = (
                td_errors.abs()
                .detach()
                .cpu()
                .numpy()
            )

            self.replay_buffer.update_priorities(
                indices,
                priorities,
            )

        # -----------------------------------------------------------------
        # Target-network update
        # -----------------------------------------------------------------

        if self.use_soft_update:

            self.soft_update_target()

        elif (
            self.steps_done
            % self.target_update_freq
            == 0
        ):

            self.hard_update_target()

        self.steps_done += 1

        return float(
            loss.item()
        )


    def save(
        self,
        filepath,
    ):
        """
        Save complete DQN checkpoint.
        """

        parent = os.path.dirname(
            filepath
        )

        if parent:

            os.makedirs(
                parent,
                exist_ok=True,
            )

        torch.save(
            {
                "policy_net":
                    self.policy_net.state_dict(),

                "target_net":
                    self.target_net.state_dict(),

                "optimizer":
                    self.optimizer.state_dict(),

                "epsilon":
                    self.epsilon,

                "steps_done":
                    self.steps_done,

                "episodes_done":
                    self.episodes_done,
            },
            filepath,
        )

        print(
            f"Model saved to {filepath}"
        )


    def load(
        self,
        filepath,
    ):
        """
        Load a previously saved DQN checkpoint.
        """

        checkpoint = torch.load(
            filepath,
            map_location=self.device,
        )

        self.policy_net.load_state_dict(
            checkpoint[
                "policy_net"
            ]
        )

        self.target_net.load_state_dict(
            checkpoint[
                "target_net"
            ]
        )

        self.optimizer.load_state_dict(
            checkpoint[
                "optimizer"
            ]
        )

        self.epsilon = checkpoint.get(
            "epsilon",
            self.epsilon,
        )

        self.steps_done = checkpoint.get(
            "steps_done",
            0,
        )

        self.episodes_done = checkpoint.get(
            "episodes_done",
            0,
        )

        print(
            f"Model loaded from {filepath}"
        )


# =========================================================================
# Training loop
# =========================================================================

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
    action_repeat=1,
):
    """
    Train the humanoid DQN agent.

    Parameters
    ----------
    env
        HumanoidWalkEnv.

    agent
        DQNAgent.

    num_episodes : int
        Number of training episodes.

    max_steps_per_episode : int
        Maximum environment steps per episode.

    save_freq : int
        Save checkpoint every N episodes.

    log_freq : int
        Print statistics every N episodes.

    save_dir : str
        Checkpoint/statistics directory.

    pose_library : list or None
        Optional initial-pose vectors.

    prefill_rollouts : int
        Number of warm-start replay-buffer rollouts.

    prefill_steps : int
        Maximum steps for each replay prefill rollout.

    action_repeat : int
        Number of environment steps for each selected action.

    Returns
    -------
    episode_rewards : list
    episode_lengths : list
    losses : list
    """

    os.makedirs(
        save_dir,
        exist_ok=True,
    )

    episode_rewards = []
    episode_lengths = []
    losses = []

    print()
    print("=" * 60)

    print(
        f"Starting Training: "
        f"{num_episodes} episodes"
    )

    print("=" * 60)
    print()

    # ---------------------------------------------------------------------
    # Optional replay-buffer prefill
    # ---------------------------------------------------------------------

    if (
        prefill_rollouts > 0
        and pose_library is not None
        and len(pose_library) > 0
    ):

        print(
            "Prefilling replay buffer with "
            f"{prefill_rollouts} warm-start rollouts "
            f"x {prefill_steps} steps..."
        )

        added = (
            _prefill_replay_with_rollouts(
                env,
                agent,
                pose_library,
                num_rollouts=prefill_rollouts,
                steps_per_rollout=prefill_steps,
            )
        )

        print(
            f"Prefill complete: "
            f"{added} transitions added."
        )

        print(
            "Current buffer size:",
            len(
                agent.replay_buffer
            ),
        )

        print()

    # ---------------------------------------------------------------------
    # Episodes
    # ---------------------------------------------------------------------

    for episode in range(
        num_episodes
    ):

        # -------------------------------------------------------------
        # Select starting pose
        # -------------------------------------------------------------

        if (
            pose_library is not None
            and len(pose_library) > 0
        ):

            pose_index = (
                np.random.randint(
                    len(pose_library)
                )
            )

            initial_pose = (
                pose_library[
                    pose_index
                ]
            )

        else:

            initial_pose = None

        state, _ = env.reset(
            initial_pose=initial_pose
        )

        episode_reward = 0.0
        episode_loss = []

        step = 0

        # -------------------------------------------------------------
        # Environment interaction
        # -------------------------------------------------------------

        while (
            step
            < max_steps_per_episode
        ):

            action = (
                agent.select_action(
                    state
                )
            )

            accumulated_reward = 0.0

            repeated_terminated = False
            repeated_truncated = False

            # ---------------------------------------------------------
            # Optional action repeat
            # ---------------------------------------------------------

            for _ in range(
                max(
                    1,
                    action_repeat,
                )
            ):

                (
                    next_state,
                    reward,
                    terminated,
                    truncated,
                    _,
                ) = env.step(
                    action
                )

                done = (
                    terminated
                    or truncated
                )

                agent.store_transition(
                    state,
                    action,
                    reward,
                    next_state,
                    done,
                )

                loss = (
                    agent.train_step()
                )

                if loss is not None:

                    episode_loss.append(
                        loss
                    )

                accumulated_reward += (
                    float(
                        reward
                    )
                )

                state = (
                    next_state
                )

                step += 1

                if (
                    done
                    or step
                    >= max_steps_per_episode
                ):

                    repeated_terminated = (
                        terminated
                    )

                    repeated_truncated = (
                        truncated
                    )

                    break

            episode_reward += (
                accumulated_reward
            )

            if (
                repeated_terminated
                or repeated_truncated
                or step
                >= max_steps_per_episode
            ):
                break

        # -------------------------------------------------------------
        # Episode completed
        # -------------------------------------------------------------

        agent.update_epsilon()

        agent.episodes_done += 1

        episode_rewards.append(
            float(
                episode_reward
            )
        )

        episode_lengths.append(
            int(
                step
            )
        )

        if episode_loss:

            losses.append(
                float(
                    np.mean(
                        episode_loss
                    )
                )
            )

        # -------------------------------------------------------------
        # Logging
        # -------------------------------------------------------------

        if (
            episode + 1
        ) % log_freq == 0:

            avg_reward = np.mean(
                episode_rewards[
                    -log_freq:
                ]
            )

            avg_length = np.mean(
                episode_lengths[
                    -log_freq:
                ]
            )

            if losses:

                avg_loss = np.mean(
                    losses[
                        -log_freq:
                    ]
                )

            else:

                avg_loss = 0.0

            print(
                f"Episode "
                f"{episode + 1}/"
                f"{num_episodes}"
            )

            print(
                f"  Avg Reward: "
                f"{avg_reward:.2f}"
            )

            print(
                f"  Avg Length: "
                f"{avg_length:.1f}"
            )

            print(
                f"  Avg Loss: "
                f"{avg_loss:.6f}"
            )

            print(
                f"  Epsilon: "
                f"{agent.epsilon:.4f}"
            )

            print(
                "  Buffer Size:",
                len(
                    agent.replay_buffer
                ),
            )

            print()

        # -------------------------------------------------------------
        # Periodic checkpoint
        # -------------------------------------------------------------

        if (
            episode + 1
        ) % save_freq == 0:

            checkpoint_path = (
                os.path.join(
                    save_dir,
                    f"dqn_episode_"
                    f"{episode + 1}.pth",
                )
            )

            agent.save(
                checkpoint_path
            )

            stats = {
                "episode_rewards": [
                    float(value)
                    for value
                    in episode_rewards
                ],

                "episode_lengths": [
                    int(value)
                    for value
                    in episode_lengths
                ],

                "losses": [
                    float(value)
                    for value
                    in losses
                ],
            }

            stats_path = os.path.join(
                save_dir,
                f"training_stats_"
                f"{episode + 1}.json",
            )

            with open(
                stats_path,
                "w",
                encoding="utf-8",
            ) as file:

                json.dump(
                    stats,
                    file,
                    indent=2,
                )

            print(
                "Training statistics saved to:",
                stats_path,
            )

    print()
    print("=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print()

    return (
        episode_rewards,
        episode_lengths,
        losses,
    )


# =========================================================================
# Training plots
# =========================================================================

def save_training_plots(
    rewards,
    lengths,
    losses,
    save_dir="checkpoints",
):
    """
    Save training curves.

    IMPORTANT:
    matplotlib is imported only here, so importing DQNAgent/train
    does not execute plotting code.
    """

    try:

        import matplotlib

        # Non-interactive backend so the training script does not
        # block waiting for graph windows to be closed.
        matplotlib.use(
            "Agg"
        )

        import matplotlib.pyplot as plt

    except ImportError:

        print(
            "[WARNING] matplotlib is not installed; "
            "training plots will not be generated."
        )

        return

    os.makedirs(
        save_dir,
        exist_ok=True,
    )

    # ---------------------------------------------------------------------
    # Reward
    # ---------------------------------------------------------------------

    if len(rewards) > 0:

        plt.figure(
            figsize=(10, 5)
        )

        plt.plot(
            rewards,
            label="Episode Reward",
        )

        plt.title(
            "Reward Curve over Episodes"
        )

        plt.xlabel(
            "Episode"
        )

        plt.ylabel(
            "Reward"
        )

        plt.grid(
            True
        )

        plt.legend()

        plt.tight_layout()

        output_path = os.path.join(
            save_dir,
            "reward_curve.png",
        )

        plt.savefig(
            output_path
        )

        plt.close()

        print(
            "Saved:",
            output_path,
        )

    # ---------------------------------------------------------------------
    # Episode lengths
    # ---------------------------------------------------------------------

    if len(lengths) > 0:

        plt.figure(
            figsize=(10, 5)
        )

        plt.plot(
            lengths,
            label="Episode Length",
        )

        plt.title(
            "Episode Length over Episodes"
        )

        plt.xlabel(
            "Episode"
        )

        plt.ylabel(
            "Length"
        )

        plt.grid(
            True
        )

        plt.legend()

        plt.tight_layout()

        output_path = os.path.join(
            save_dir,
            "length_curve.png",
        )

        plt.savefig(
            output_path
        )

        plt.close()

        print(
            "Saved:",
            output_path,
        )

    # ---------------------------------------------------------------------
    # Training loss
    # ---------------------------------------------------------------------

    if len(losses) > 0:

        plt.figure(
            figsize=(10, 5)
        )

        plt.plot(
            losses,
            label="Average Episode Loss",
        )

        plt.title(
            "DQN Training Loss"
        )

        plt.xlabel(
            "Training Episode"
        )

        plt.ylabel(
            "Loss"
        )

        plt.grid(
            True
        )

        plt.legend()

        plt.tight_layout()

        output_path = os.path.join(
            save_dir,
            "loss_curve.png",
        )

        plt.savefig(
            output_path
        )

        plt.close()

        print(
            "Saved:",
            output_path,
        )


# =========================================================================
# Command-line interface
# =========================================================================

def build_argument_parser():
    """
    Create command-line argument parser.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Train DQN/Dueling-DQN for "
            "image-conditioned humanoid locomotion."
        )
    )

    parser.add_argument(
        "--pose-dir",
        type=str,
        default="data/output_image",
        help=(
            "Directory containing "
            "*_pose_vector.npy files."
        ),
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=150,
        help="Number of training episodes.",
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=2000,
        help=(
            "Maximum environment steps "
            "per episode."
        ),
    )

    parser.add_argument(
        "--save-freq",
        type=int,
        default=30,
        help=(
            "Checkpoint frequency in episodes."
        ),
    )

    parser.add_argument(
        "--log-freq",
        type=int,
        default=5,
        help=(
            "Logging frequency in episodes."
        ),
    )

    parser.add_argument(
        "--prefill-rollouts",
        type=int,
        default=300,
        help=(
            "Warm-start rollouts used to "
            "prefill replay memory."
        ),
    )

    parser.add_argument(
        "--prefill-steps",
        type=int,
        default=50,
        help=(
            "Maximum steps per replay "
            "prefill rollout."
        ),
    )

    parser.add_argument(
        "--action-repeat",
        type=int,
        default=1,
        help=(
            "Environment steps for each "
            "selected DQN action."
        ),
    )

    # Kept for compatibility with your existing training command.
    parser.add_argument(
        "--dueling",
        action="store_true",
        help=(
            "Use Dueling DQN. "
            "Dueling DQN is already the default."
        ),
    )

    parser.add_argument(
        "--standard-dqn",
        action="store_true",
        help=(
            "Use standard DQN instead of "
            "Dueling DQN."
        ),
    )

    parser.add_argument(
        "--prioritized",
        action="store_true",
        help=(
            "Use prioritized replay memory."
        ),
    )

    parser.add_argument(
        "--save-dir",
        type=str,
        default="checkpoints",
        help=(
            "Directory for checkpoints, "
            "statistics, and plots."
        ),
    )

    return parser


# =========================================================================
# Standalone training entry point
# =========================================================================

def main():

    parser = (
        build_argument_parser()
    )

    args = parser.parse_args()

    print("=" * 70)
    print("DQN HUMANOID TRAINING")
    print("=" * 70)

    print(
        "Initializing environment..."
    )

    os.makedirs(
        args.save_dir,
        exist_ok=True,
    )

    env = HumanoidWalkEnv(
        render_mode=None,
        max_episode_steps=args.max_steps,
    )

    try:

        use_dueling = (
            not args.standard_dqn
        )

        print(
            "Network:",
            (
                "Dueling DQN"
                if use_dueling
                else "Standard DQN"
            ),
        )

        print(
            "Prioritized replay:",
            args.prioritized,
        )

        # -----------------------------------------------------------------
        # Agent
        # -----------------------------------------------------------------

        agent = DQNAgent(
            env=env,
            use_dueling=use_dueling,
            use_prioritized=args.prioritized,
            learning_rate=1e-4,
            gamma=0.99,
            epsilon_start=1.0,
            epsilon_end=0.05,
            epsilon_decay=0.985,
            buffer_capacity=30000,
            batch_size=32,
        )

        # -----------------------------------------------------------------
        # Pose library
        # -----------------------------------------------------------------

        pose_library = (
            _load_pose_vectors_from_dir(
                args.pose_dir,
                target_dim=env.num_joints,
            )
        )

        if pose_library:

            print(
                f"Loaded {len(pose_library)} "
                f"pose vector(s) from "
                f"{args.pose_dir}"
            )

        else:

            print(
                "No pose vectors found in "
                f"{args.pose_dir}; "
                "training will use the "
                "environment default pose."
            )

        # -----------------------------------------------------------------
        # Train
        # -----------------------------------------------------------------

        rewards, lengths, losses = train(
            env=env,
            agent=agent,
            num_episodes=args.episodes,
            max_steps_per_episode=args.max_steps,
            save_freq=args.save_freq,
            log_freq=args.log_freq,
            save_dir=args.save_dir,
            pose_library=(
                pose_library
                if pose_library
                else None
            ),
            prefill_rollouts=(
                args.prefill_rollouts
                if pose_library
                else 0
            ),
            prefill_steps=args.prefill_steps,
            action_repeat=max(
                1,
                args.action_repeat,
            ),
        )

        # -----------------------------------------------------------------
        # Final checkpoint
        # -----------------------------------------------------------------

        final_model_path = os.path.join(
            args.save_dir,
            "dqn_final.pth",
        )

        agent.save(
            final_model_path
        )

        # -----------------------------------------------------------------
        # Final statistics file
        # -----------------------------------------------------------------

        final_stats_path = os.path.join(
            args.save_dir,
            "training_stats_final.json",
        )

        final_stats = {
            "episode_rewards": [
                float(value)
                for value
                in rewards
            ],

            "episode_lengths": [
                int(value)
                for value
                in lengths
            ],

            "losses": [
                float(value)
                for value
                in losses
            ],

            "episodes": int(
                len(rewards)
            ),

            "final_epsilon": float(
                agent.epsilon
            ),

            "steps_done": int(
                agent.steps_done
            ),
        }

        with open(
            final_stats_path,
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                final_stats,
                file,
                indent=2,
            )

        print(
            "Final statistics saved to:",
            final_stats_path,
        )

        # -----------------------------------------------------------------
        # Plots
        # -----------------------------------------------------------------

        save_training_plots(
            rewards=rewards,
            lengths=lengths,
            losses=losses,
            save_dir=args.save_dir,
        )

        print()
        print("=" * 70)
        print("TRAINING COMPLETED")
        print("=" * 70)

        print(
            "Final model:",
            final_model_path,
        )

        print(
            "Episodes:",
            len(rewards),
        )

        if rewards:

            print(
                "Average reward (last 10):",
                f"{np.mean(rewards[-10:]):.2f}",
            )

        if lengths:

            print(
                "Average length (last 10):",
                f"{np.mean(lengths[-10:]):.1f}",
            )

        print(
            "Final epsilon:",
            f"{agent.epsilon:.4f}",
        )

    finally:

        env.close()


# =========================================================================
# CRITICAL:
# Nothing below this runs when the module is imported by main.py.
# =========================================================================

if __name__ == "__main__":
    main()