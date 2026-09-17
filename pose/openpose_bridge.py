import json
import os
import subprocess
import tempfile

import numpy as np


PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

WORKER_PATH = os.path.join(
    PROJECT_ROOT,
    "pose",
    "openpose_worker.py",
)


def extract_pose_with_openpose(
    image_path,
    timeout=300,
):
    """
    Run the Python-3.7 OpenPose worker from the Python-3.10
    RL/simulation environment.

    Pipeline
    --------
    Python 3.10
        -> subprocess
        -> Python 3.7
        -> OpenPose BODY_25
        -> primary-person selection
        -> 8-joint pose vector
        -> temporary JSON
        -> Python 3.10

    Parameters
    ----------
    image_path : str
        Path to input image.

    timeout : int
        Maximum worker execution time in seconds.

    Returns
    -------
    result : dict
        Full worker result.

        Includes:

        result["pose_vector_np"]
            np.ndarray with shape (8,)

        result["actuator_order"]
            names of the 8 simulated joints
    """

    image_path = os.path.abspath(
        image_path
    )

    if not os.path.exists(image_path):
        raise FileNotFoundError(
            f"Input image not found: {image_path}"
        )

    if not os.path.exists(WORKER_PATH):
        raise FileNotFoundError(
            f"OpenPose worker not found: {WORKER_PATH}"
        )

    # ---------------------------------------------------------
    # Create temporary JSON output file
    # ---------------------------------------------------------

    fd, output_path = tempfile.mkstemp(
        prefix="openpose_pose_",
        suffix=".json",
    )

    os.close(fd)

    command = [
        "py",
        "-3.7",
        WORKER_PATH,
        "--image",
        image_path,
        "--output",
        output_path,
    ]

    try:
        # -----------------------------------------------------
        # Run the Python-3.7 OpenPose process
        # -----------------------------------------------------

        process = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        if process.returncode != 0:
            raise RuntimeError(
                "OpenPose worker failed.\n\n"
                f"STDOUT:\n{process.stdout}\n\n"
                f"STDERR:\n{process.stderr}"
            )

        if not os.path.exists(output_path):
            raise RuntimeError(
                "OpenPose worker completed but did not "
                "produce its JSON output."
            )

        # -----------------------------------------------------
        # Load worker output
        # -----------------------------------------------------

        with open(
            output_path,
            "r",
            encoding="utf-8",
        ) as file:
            result = json.load(file)

        if not result.get(
            "success",
            False,
        ):
            raise RuntimeError(
                "OpenPose worker reported failure: "
                + str(
                    result.get(
                        "error",
                        "Unknown error",
                    )
                )
            )

        # -----------------------------------------------------
        # Validate pose vector
        # -----------------------------------------------------

        pose_vector = np.asarray(
            result["pose_vector"],
            dtype=np.float32,
        )

        if pose_vector.shape != (8,):
            raise ValueError(
                "Expected an 8-joint pose vector, "
                f"received shape {pose_vector.shape}"
            )

        if not np.all(
            np.isfinite(
                pose_vector
            )
        ):
            raise ValueError(
                "OpenPose worker returned non-finite "
                "joint angles."
            )

        result[
            "pose_vector_np"
        ] = pose_vector

        result[
            "worker_stdout"
        ] = process.stdout

        return result

    except subprocess.TimeoutExpired as exc:
        raise TimeoutError(
            "OpenPose worker exceeded the "
            f"{timeout}-second timeout."
        ) from exc

    finally:
        # -----------------------------------------------------
        # Always remove temporary JSON
        # -----------------------------------------------------

        if os.path.exists(
            output_path
        ):
            try:
                os.remove(
                    output_path
                )
            except OSError:
                pass