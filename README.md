# Image-to-Pose Reinforcement Learning for Humanoid Locomotion

A computer vision and deep reinforcement learning project that converts a human pose from an image into an initial pose for a simulated humanoid and trains a DQN-based controller in PyBullet.

## Project Overview

The project integrates:

- OpenPose BODY_25 human pose estimation
- Multi-person detection and primary-person selection
- 2D skeleton-to-humanoid pose conversion
- Custom PyBullet humanoid simulation
- Gymnasium-compatible reinforcement learning environment
- DQN and Dueling DQN
- Experience replay and target networks
- Torque-based joint control
- Pose-conditioned environment resets
- Checkpointing and policy evaluation

## Pipeline

```text
Input Image
    -> OpenPose BODY_25
    -> Multi-Person Detection
    -> Primary-Person Selection
    -> 8-Joint Pose Vector
    -> PyBullet Humanoid
    -> Gymnasium Environment
    -> DQN / Dueling DQN
    -> Torque-Based Control
```

When multiple people are detected, the person with the largest bounding-box area is selected as the primary subject.

## Humanoid Pose Representation

The BODY_25 skeleton is converted into an 8-dimensional pose vector corresponding to:

1. `right_hip`
2. `right_knee`
3. `left_hip`
4. `left_knee`
5. `left_shoulder`
6. `left_elbow`
7. `right_shoulder`
8. `right_elbow`

The generated joint angles are constrained by the limits defined in the humanoid URDF before being applied to the simulator.

The image-to-humanoid mapping uses 2D skeleton geometry rather than full 3D inverse kinematics.

## PyBullet Environment

The custom Gymnasium environment contains:

- 8 dynamically discovered controllable joints
- 29-dimensional observation space
- 5 discrete torque choices per joint
- torque-based control
- pose-conditioned resets
- fall detection
- episode termination and truncation

The action space is:

```text
MultiDiscrete([5, 5, 5, 5, 5, 5, 5, 5])
```

## Reinforcement Learning

The project implements:

- DQN
- Dueling DQN
- experience replay
- optional prioritized replay
- target-network updates
- epsilon-greedy exploration
- gradient clipping
- checkpoint saving
- training-statistics logging

The verified end-to-end training run used Dueling DQN.

## Reward Function

The current locomotion reward combines:

```text
forward velocity
+ alive / survival bonus
- energy expenditure penalty
```

Fall detection is handled separately through environment termination conditions.

The current trained policy is experimental and does not guarantee a stable walking gait.

## OpenPose Integration

OpenPose BODY_25 is used to extract human skeletons.

For multi-person images:

1. OpenPose detects all visible people.
2. A bounding box is calculated for each detected person.
3. The person with the largest bounding-box area is selected.
4. BODY_25 geometry is converted into the humanoid's 8-joint pose representation.

## Dual Python Runtime

The project uses two Python runtimes because the available OpenPose Python binding was compiled specifically for Python 3.7.

### Main Runtime

Python 3.10 is used for:

- PyTorch
- PyBullet
- Gymnasium
- training
- simulation
- evaluation

### OpenPose Runtime

Python 3.7 is used for `pyopenpose`.

`pose/openpose_worker.py` performs OpenPose inference using Python 3.7.

`pose/openpose_bridge.py` allows the main Python 3.10 application to launch the worker and receive the extracted pose information.

## Project Structure

```text
deep-reinforcement-humanoid/
|
|-- agent/
|   |-- network.py
|   |-- replay.py
|   |-- train_dqn.py
|   |-- test_trained_agent.py
|   `-- generate_pose_library.py
|
|-- env/
|   |-- __init__.py
|   |-- humanoid.urdf
|   `-- walk_env.py
|
|-- pose/
|   |-- angles.py
|   |-- keypoints.py
|   |-- openpose_bridge.py
|   `-- openpose_worker.py
|
|-- data/
|   `-- input_images/
|
|-- main.py
|-- test_env.py
|-- test_pose.py
|-- requirements.txt
|-- requirements-openpose.txt
|-- .gitignore
`-- README.md
```

## Verified Configuration

Main runtime:

```text
Python       3.10.7
NumPy        1.26.4
OpenCV       4.10.0.84
PyBullet     3.2.6
Gymnasium    0.29.1
PyTorch      2.14.0
Matplotlib   3.10.9
```

OpenPose worker:

```text
Python       3.7
NumPy        1.21.6
OpenCV       4.5.5.64
```

## OpenPose Setup

The verified Windows configuration expects OpenPose at:

```text
C:\openpose
```

The Python binding used during development is:

```text
C:\openpose\build\python\openpose\Release\pyopenpose.cp37-win_amd64.pyd
```

Required DLL locations include:

```text
C:\openpose\build\x64\Release
C:\openpose\build\bin
C:\openpose\3rdparty\windows\caffe\bin
C:\openpose\3rdparty\windows\opencv\x64\vc15\bin
```

If OpenPose is installed elsewhere, update `OPENPOSE_DIR` in `pose/keypoints.py`.

## Installation

Create the main Python 3.10 environment:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Install the Python 3.7 OpenPose worker dependencies:

```powershell
py -3.7 -m pip install -r requirements-openpose.txt
```

`pyopenpose` itself is provided by the local OpenPose build and is not installed through PyPI.

## Validate the Complete Pipeline

Run:

```powershell
python .\test_env.py --no-render
```

The integration test validates:

- custom humanoid URDF
- dynamic joint discovery
- Gymnasium API
- torque-based control
- OpenPose BODY_25
- multi-person selection
- 8-joint pose conversion
- image-conditioned reset
- PyBullet simulation
- trained-model loading when a local checkpoint is available

A successful run ends with:

```text
ALL CORE TESTS PASSED
```

## Generate a Pose Library

```powershell
python .\agent\generate_pose_library.py --image-dir .\data\input_images --output .\data\poses\pose_library.pkl
```

Generated pose libraries are excluded from Git.

## Run the Complete Pipeline

```powershell
python .\main.py
```

The current pipeline performs:

```text
Input Image
-> OpenPose BODY_25
-> Primary-Person Selection
-> 8-Joint Pose Conversion
-> PyBullet Humanoid Reset
-> Dueling DQN Training
-> Checkpoint Saving
```

The current `main.py` configuration runs 150 training episodes.

## Evaluate a Trained Model

Visual evaluation:

```powershell
python .\agent\test_trained_agent.py --model .\checkpoints\dqn_final.pth --episodes 5
```

Headless evaluation:

```powershell
python .\agent\test_trained_agent.py --model .\checkpoints\dqn_final.pth --episodes 5 --no-render
```

Generated checkpoints are stored under `checkpoints/` and are excluded from Git.

## Verified Example

Using `data/input_images/sample2.jpeg`, the pipeline successfully:

- detected 4 people with OpenPose BODY_25
- selected the person with the largest bounding-box area
- generated an 8-joint pose vector
- matched the pose ordering with the PyBullet actuator ordering
- applied the image-derived pose without joint-limit clipping
- initialized the PyBullet simulation
- completed the full integration test

## Current Limitations

- 2D pose geometry rather than full 3D reconstruction
- local Python 3.7 OpenPose dependency
- limited humanoid degrees of freedom
- no explicit torso-upright reward term
- locomotion quality depends strongly on reward design and training hyperparameters
- the current trained policy does not guarantee stable walking

## Future Work

- explicit upright-orientation reward
- body-height stability reward
- bounded target-speed reward
- stronger fall penalties
- action-smoothness regularization
- larger pose libraries
- longer training schedules
- PPO or SAC comparison
- 3D human pose estimation
- inverse-kinematics initialization
- quantitative gait evaluation

## Technologies

Python, PyTorch, OpenPose, OpenCV, PyBullet, Gymnasium, NumPy, Matplotlib, Deep Reinforcement Learning, DQN and Dueling DQN.
