"""
Generate a reusable library of humanoid initial poses from images.

The script runs under the main Python 3.10 environment.

OpenPose itself is executed through pose.openpose_bridge, which launches
the Python 3.7 OpenPose worker and returns the final 8-joint pose vector.
"""

import argparse
import os
import pickle
import sys
from pathlib import Path

import numpy as np


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


from pose.openpose_bridge import extract_pose_with_openpose


SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tiff",
}


def generate_pose_library(
    image_dir,
    output_path="data/poses/pose_library.pkl",
    target_joints=8,
):
    """
    Generate a library of initial humanoid poses from images.

    Parameters
    ----------
    image_dir : str
        Directory containing input images.

    output_path : str
        Output .pkl path.

    target_joints : int
        Desired number of pose values. The current humanoid uses 8.

    Returns
    -------
    list[np.ndarray]
        Extracted pose vectors.
    """

    image_dir = Path(
        image_dir
    )

    if not image_dir.exists():
        raise FileNotFoundError(
            f"Image directory not found: {image_dir}"
        )

    image_paths = sorted(
        [
            path
            for path in image_dir.iterdir()
            if (
                path.is_file()
                and path.suffix.lower()
                in SUPPORTED_EXTENSIONS
            )
        ]
    )

    if not image_paths:
        print(
            f"No supported images found in {image_dir}"
        )
        return []

    print(
        f"Found {len(image_paths)} image(s)"
    )

    print(
        "Processing images through OpenPose..."
    )

    print()

    pose_library = []
    failed_images = []

    for index, image_path in enumerate(
        image_paths,
        start=1,
    ):

        try:

            print(
                f"[{index}/{len(image_paths)}] "
                f"Processing: {image_path.name}"
            )

            result = extract_pose_with_openpose(
                str(image_path)
            )

            theta = np.asarray(
                result["pose_vector_np"],
                dtype=np.float32,
            )

            if theta.shape != (8,):
                raise ValueError(
                    "OpenPose bridge returned "
                    f"unexpected pose shape {theta.shape}"
                )

            if not np.all(
                np.isfinite(theta)
            ):
                raise ValueError(
                    "Pose contains non-finite values"
                )

            # Keep target_joints argument for compatibility.
            if target_joints < len(theta):

                theta = theta[
                    :target_joints
                ]

            elif target_joints > len(theta):

                theta = np.pad(
                    theta,
                    (
                        0,
                        target_joints
                        - len(theta),
                    ),
                    mode="constant",
                )

            pose_library.append(
                theta
            )

            print(
                "  [OK] People detected:",
                result["people_detected"],
            )

            print(
                "  [OK] Selected person:",
                result[
                    "selected_person_number"
                ],
            )

            print(
                "  [OK] Pose:",
                theta,
            )

            print()

        except Exception as exc:

            print(
                f"  [FAILED] {exc}"
            )

            failed_images.append(
                str(image_path)
            )

            print()

    print("=" * 70)
    print("POSE EXTRACTION SUMMARY")
    print("=" * 70)

    print(
        "Total images:",
        len(image_paths),
    )

    print(
        "Successful:",
        len(pose_library),
    )

    print(
        "Failed:",
        len(failed_images),
    )

    if failed_images:

        print()
        print("Failed images:")

        for image in failed_images:
            print(
                "  -",
                image,
            )

    if not pose_library:

        print()
        print(
            "No poses extracted. "
            "Library was not saved."
        )

        return []

    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ---------------------------------------------------------------------
    # Pickle version
    # ---------------------------------------------------------------------

    with open(
        output_path,
        "wb",
    ) as file:

        pickle.dump(
            pose_library,
            file,
        )

    print()
    print(
        "Pose library saved to:",
        output_path,
    )

    # ---------------------------------------------------------------------
    # NumPy version
    # ---------------------------------------------------------------------

    numpy_path = (
        output_path.with_suffix(
            ".npy"
        )
    )

    np.save(
        numpy_path,
        np.asarray(
            pose_library,
            dtype=np.float32,
        ),
    )

    print(
        "NumPy copy saved to:",
        numpy_path,
    )

    print(
        "Number of poses:",
        len(pose_library),
    )

    return pose_library


def load_pose_library(
    library_path
):
    """
    Load an existing .pkl or .npy pose library.
    """

    library_path = Path(
        library_path
    )

    suffix = (
        library_path.suffix.lower()
    )

    if suffix == ".pkl":

        with open(
            library_path,
            "rb",
        ) as file:

            return pickle.load(
                file
            )

    if suffix == ".npy":

        return np.load(
            library_path
        ).tolist()

    raise ValueError(
        "Library must be a .pkl or .npy file"
    )


def visualize_pose_library(
    library_path
):
    """
    Print basic pose-library statistics.
    """

    poses = load_pose_library(
        library_path
    )

    poses_array = np.asarray(
        poses,
        dtype=np.float32,
    )

    if (
        poses_array.ndim != 2
        or len(poses_array) == 0
    ):
        raise ValueError(
            "Pose library is empty or malformed"
        )

    print()
    print("=" * 70)
    print("POSE LIBRARY STATISTICS")
    print("=" * 70)

    print(
        "Library path:",
        library_path,
    )

    print(
        "Number of poses:",
        len(poses_array),
    )

    print(
        "Pose dimension:",
        poses_array.shape[1],
    )

    print()
    print(
        f"{'Joint':<10}"
        f"{'Mean':<12}"
        f"{'Std':<12}"
        f"{'Min':<12}"
        f"{'Max':<12}"
    )

    print("-" * 58)

    for joint_index in range(
        poses_array.shape[1]
    ):

        values = poses_array[
            :,
            joint_index,
        ]

        print(
            f"{joint_index:<10}"
            f"{np.mean(values):<12.3f}"
            f"{np.std(values):<12.3f}"
            f"{np.min(values):<12.3f}"
            f"{np.max(values):<12.3f}"
        )

    print()
    print(
        "Sample poses:"
    )

    for index, pose in enumerate(
        poses_array[:5],
        start=1,
    ):

        print(
            f"  Pose {index}: {pose}"
        )


def add_synthetic_poses(
    pose_library,
    num_synthetic=10,
    noise_std=0.3,
):
    """
    Add synthetic poses by perturbing existing pose vectors.
    """

    if not pose_library:

        print(
            "Pose library is empty; "
            "synthetic poses cannot be generated."
        )

        return pose_library

    poses_array = np.asarray(
        pose_library,
        dtype=np.float32,
    )

    synthetic_poses = []

    for _ in range(
        num_synthetic
    ):

        base_pose = poses_array[
            np.random.randint(
                len(poses_array)
            )
        ]

        noise = np.random.normal(
            loc=0.0,
            scale=noise_std,
            size=base_pose.shape,
        )

        synthetic_pose = (
            base_pose
            + noise
        )

        synthetic_pose = np.clip(
            synthetic_pose,
            -np.pi,
            np.pi,
        ).astype(
            np.float32
        )

        synthetic_poses.append(
            synthetic_pose
        )

    augmented_library = (
        list(pose_library)
        + synthetic_poses
    )

    print(
        f"Added {num_synthetic} synthetic poses"
    )

    print(
        "Total poses:",
        len(augmented_library),
    )

    return augmented_library


def save_pose_library(
    pose_library,
    output_path,
):
    """
    Save both pickle and NumPy representations.
    """

    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        output_path,
        "wb",
    ) as file:

        pickle.dump(
            pose_library,
            file,
        )

    numpy_path = (
        output_path.with_suffix(
            ".npy"
        )
    )

    np.save(
        numpy_path,
        np.asarray(
            pose_library,
            dtype=np.float32,
        ),
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Generate a humanoid pose library "
            "from input images."
        )
    )

    parser.add_argument(
        "--image-dir",
        type=str,
        default="data/input_images",
        help=(
            "Directory containing input images."
        ),
    )

    parser.add_argument(
        "--output",
        type=str,
        default=(
            "data/poses/"
            "pose_library.pkl"
        ),
        help="Output pose-library path.",
    )

    parser.add_argument(
        "--joints",
        type=int,
        default=8,
        help=(
            "Target pose dimension. "
            "Current humanoid uses 8."
        ),
    )

    parser.add_argument(
        "--visualize",
        type=str,
        default=None,
        help=(
            "Visualize an existing .pkl "
            "or .npy pose library."
        ),
    )

    parser.add_argument(
        "--add-synthetic",
        type=int,
        default=0,
        help=(
            "Number of synthetic perturbed "
            "poses to add."
        ),
    )

    args = parser.parse_args()

    if args.visualize:

        visualize_pose_library(
            args.visualize
        )

    else:

        pose_library = generate_pose_library(
            image_dir=args.image_dir,
            output_path=args.output,
            target_joints=args.joints,
        )

        if (
            args.add_synthetic > 0
            and pose_library
        ):

            pose_library = add_synthetic_poses(
                pose_library,
                num_synthetic=(
                    args.add_synthetic
                ),
            )

            save_pose_library(
                pose_library,
                args.output,
            )

            print()
            print(
                "[OK] Augmented library saved"
            )

        if pose_library:

            visualize_pose_library(
                args.output
            )