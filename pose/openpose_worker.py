import argparse
import json
import os
import sys


# -------------------------------------------------------------------------
# Make project root importable when this file is executed directly:
#
# py -3.7 pose\openpose_worker.py
# -------------------------------------------------------------------------

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


from pose.keypoints import extract_keypoints
from pose.angles import skeleton_to_pose_vector


def process_image(image_path):
    """
    Run the complete OpenPose stage:

    image
      -> BODY_25 skeletons
      -> largest-person selection
      -> 8-joint humanoid pose vector

    Returns a JSON-serializable dictionary.
    """

    image_path = os.path.abspath(image_path)

    if not os.path.exists(image_path):
        raise FileNotFoundError(
            "Input image does not exist: "
            + image_path
        )

    # ---------------------------------------------------------
    # OpenPose BODY_25
    # ---------------------------------------------------------

    (
        keypoints,
        _,
        selected_index,
        people_data,
    ) = extract_keypoints(
        image_path,
        return_image=False,
        verbose=False,
    )

    # ---------------------------------------------------------
    # Convert selected skeleton to humanoid pose
    # ---------------------------------------------------------

    theta_init, metadata = skeleton_to_pose_vector(
        keypoints,
        selected_index=selected_index,
        verbose=False,
    )

    selected_person = people_data[
        selected_index
    ]

    # ---------------------------------------------------------
    # Build JSON-safe output
    # ---------------------------------------------------------

    result = {
        "success": True,

        "image_path": image_path,

        "people_detected": int(
            len(people_data)
        ),

        "selected_person_index": int(
            selected_index
        ),

        "selected_person_number": int(
            selected_index + 1
        ),

        "selected_person": {
            "area": int(
                selected_person["area"]
            ),

            "width": int(
                selected_person["width"]
            ),

            "height": int(
                selected_person["height"]
            ),

            "valid_keypoints": int(
                selected_person[
                    "valid_keypoints"
                ]
            ),

            "bbox": [
                int(x)
                for x in selected_person[
                    "bbox"
                ]
            ],
        },

        "pose_vector": [
            float(x)
            for x in theta_init
        ],

        "pose_vector_length": int(
            len(theta_init)
        ),

        "actuator_order": list(
            metadata["actuator_order"]
        ),

        "segment_orientations_rad": [
            float(x)
            for x in metadata[
                "segment_orientations_rad"
            ]
        ],

        "spine_angle_rad": float(
            metadata[
                "spine_angle_rad"
            ]
        ),

        "spine_angle_deg": float(
            metadata[
                "spine_angle_deg"
            ]
        ),
    }

    return result


def main():
    parser = argparse.ArgumentParser(
        description=(
            "OpenPose BODY_25 worker for the "
            "image-to-humanoid pipeline."
        )
    )

    parser.add_argument(
        "--image",
        required=True,
        help="Path to the input image.",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Path where the worker JSON result will be written.",
    )

    args = parser.parse_args()

    output_path = os.path.abspath(
        args.output
    )

    output_directory = os.path.dirname(
        output_path
    )

    if output_directory:
        os.makedirs(
            output_directory,
            exist_ok=True,
        )

    try:

        result = process_image(
            args.image
        )

    except Exception as exc:

        result = {
            "success": False,
            "error_type": type(
                exc
            ).__name__,
            "error": str(
                exc
            ),
        }

        # Write the error JSON before re-raising.
        with open(
            output_path,
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                result,
                file,
                indent=2,
            )

        raise

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            result,
            file,
            indent=2,
        )

    print()
    print("=" * 70)
    print("OPENPOSE WORKER COMPLETED")
    print("=" * 70)

    print(
        "People detected:",
        result["people_detected"],
    )

    print(
        "Selected person:",
        result[
            "selected_person_number"
        ],
    )

    print(
        "Pose-vector length:",
        result[
            "pose_vector_length"
        ],
    )

    print(
        "Pose vector:",
        result[
            "pose_vector"
        ],
    )

    print(
        "JSON output:",
        output_path,
    )


if __name__ == "__main__":
    main()