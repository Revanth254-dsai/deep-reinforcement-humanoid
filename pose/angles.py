import numpy as np


# -------------------------------------------------------------------------
# OpenPose BODY_25 keypoint names
# -------------------------------------------------------------------------

BODY_25_NAMES = [
    "Nose",
    "Neck",
    "RShoulder",
    "RElbow",
    "RWrist",
    "LShoulder",
    "LElbow",
    "LWrist",
    "MidHip",
    "RHip",
    "RKnee",
    "RAnkle",
    "LHip",
    "LKnee",
    "LAnkle",
    "REye",
    "LEye",
    "REar",
    "LEar",
    "LBigToe",
    "LSmallToe",
    "LHeel",
    "RBigToe",
    "RSmallToe",
    "RHeel",
]


# -------------------------------------------------------------------------
# BODY_25 segments mapped to the 8 actuators in HumanoidWalkEnv
#
# Environment actuator order:
#
# 0 -> right_hip
# 1 -> right_knee
# 2 -> left_hip
# 3 -> left_knee
# 4 -> left_shoulder
# 5 -> left_elbow
# 6 -> right_shoulder
# 7 -> right_elbow
# -------------------------------------------------------------------------

ACTUATOR_PAIRS = [
    ("right_hip", 9, 10),       # RHip -> RKnee
    ("right_knee", 10, 11),     # RKnee -> RAnkle

    ("left_hip", 12, 13),       # LHip -> LKnee
    ("left_knee", 13, 14),      # LKnee -> LAnkle

    ("left_shoulder", 5, 6),    # LShoulder -> LElbow
    ("left_elbow", 6, 7),       # LElbow -> LWrist

    ("right_shoulder", 2, 3),   # RShoulder -> RElbow
    ("right_elbow", 3, 4),      # RElbow -> RWrist
]


# -------------------------------------------------------------------------
# Spine information
#
# The current humanoid URDF has no controllable spine actuator.
# We therefore retain the spine orientation only as metadata.
# -------------------------------------------------------------------------

SPINE_PAIR = (1, 8)  # Neck -> MidHip


# -------------------------------------------------------------------------
# Humanoid joint limits
#
# IMPORTANT:
# These values must match env/humanoid.urdf.
#
# Order matches ACTUATOR_PAIRS.
# -------------------------------------------------------------------------

HUMANOID_JOINT_LIMITS = np.array(
    [
        [-1.5, 1.5],    # right_hip
        [-2.0, 0.2],    # right_knee

        [-1.5, 1.5],    # left_hip
        [-2.0, 0.2],    # left_knee

        [-2.0, 2.0],    # left_shoulder
        [-2.2, 0.2],    # left_elbow

        [-2.0, 2.0],    # right_shoulder
        [-2.2, 0.2],    # right_elbow
    ],
    dtype=np.float32,
)


# -------------------------------------------------------------------------
# Segment-angle helper
# -------------------------------------------------------------------------

def _segment_angle(
    skeleton,
    joint_a,
    joint_b,
):
    """
    Calculate the absolute 2D orientation of a BODY_25 segment.

    Parameters
    ----------
    skeleton : np.ndarray
        BODY_25 skeleton with shape (25, 3).

    joint_a : int
        Start keypoint index.

    joint_b : int
        End keypoint index.

    Returns
    -------
    angle : float
        Segment orientation in radians.

    details : dict or None
        Metadata describing the calculation.

    Notes
    -----
    Image coordinates use:

        +x -> right
        +y -> downward

    Therefore a vertically downward segment has an orientation
    close to +pi/2.
    """

    point_a = skeleton[
        joint_a
    ]

    point_b = skeleton[
        joint_b
    ]

    # OpenPose confidence threshold
    if (
        point_a[2] <= 0.1
        or point_b[2] <= 0.1
    ):
        return 0.0, None

    vector = (
        point_b[:2]
        - point_a[:2]
    )

    angle = float(
        np.arctan2(
            vector[1],
            vector[0],
        )
    )

    details = {
        "joint_a": BODY_25_NAMES[
            joint_a
        ],

        "joint_b": BODY_25_NAMES[
            joint_b
        ],

        "point_a": [
            float(point_a[0]),
            float(point_a[1]),
            float(point_a[2]),
        ],

        "point_b": [
            float(point_b[0]),
            float(point_b[1]),
            float(point_b[2]),
        ],

        "vector": [
            float(vector[0]),
            float(vector[1]),
        ],

        "angle_rad": angle,

        "angle_deg": float(
            np.degrees(
                angle
            )
        ),

        "status": "valid",
    }

    return angle, details


# -------------------------------------------------------------------------
# Angle normalization
# -------------------------------------------------------------------------

def _wrap_to_pi(
    angle,
):
    """
    Normalize an angle to [-pi, pi].
    """

    return (
        (
            angle
            + np.pi
        )
        % (
            2.0
            * np.pi
        )
        - np.pi
    )


# -------------------------------------------------------------------------
# Convert image segment orientations to humanoid joint angles
# -------------------------------------------------------------------------

def _segments_to_humanoid_pose(
    segment_angles,
):
    """
    Convert BODY_25 image-segment orientations into relative
    humanoid joint angles.

    Input order
    -----------
    0 -> right thigh
    1 -> right shin
    2 -> left thigh
    3 -> left shin
    4 -> left upper arm
    5 -> left forearm
    6 -> right upper arm
    7 -> right forearm

    Output order
    ------------
    0 -> right_hip
    1 -> right_knee
    2 -> left_hip
    3 -> left_knee
    4 -> left_shoulder
    5 -> left_elbow
    6 -> right_shoulder
    7 -> right_elbow

    Notes
    -----
    Hip and shoulder rotation is measured relative to the
    vertical-down neutral direction.

    Knee and elbow values represent relative flexion between
    adjacent limb segments.
    """

    a = np.asarray(
        segment_angles,
        dtype=np.float32,
    )

    if a.shape != (
        8,
    ):
        raise ValueError(
            "Expected 8 segment orientations, "
            f"received {a.shape}"
        )

    pose = np.array(
        [
            # -------------------------------------------------------------
            # RIGHT LEG
            # -------------------------------------------------------------

            # Right hip:
            # thigh relative to downward vertical
            _wrap_to_pi(
                a[0]
                - np.pi / 2.0
            ),

            # Right knee:
            # relative flexion between thigh and shin
            -abs(
                _wrap_to_pi(
                    a[1]
                    - a[0]
                )
            ),

            # -------------------------------------------------------------
            # LEFT LEG
            # -------------------------------------------------------------

            _wrap_to_pi(
                a[2]
                - np.pi / 2.0
            ),

            -abs(
                _wrap_to_pi(
                    a[3]
                    - a[2]
                )
            ),

            # -------------------------------------------------------------
            # LEFT ARM
            # -------------------------------------------------------------

            _wrap_to_pi(
                a[4]
                - np.pi / 2.0
            ),

            -abs(
                _wrap_to_pi(
                    a[5]
                    - a[4]
                )
            ),

            # -------------------------------------------------------------
            # RIGHT ARM
            # -------------------------------------------------------------

            _wrap_to_pi(
                a[6]
                - np.pi / 2.0
            ),

            -abs(
                _wrap_to_pi(
                    a[7]
                    - a[6]
                )
            ),
        ],
        dtype=np.float32,
    )

    return pose


# -------------------------------------------------------------------------
# Main conversion function
# -------------------------------------------------------------------------

def skeleton_to_pose_vector(
    keypoints_all,
    selected_index=0,
    verbose=True,
):
    """
    Convert an OpenPose BODY_25 skeleton into the 8-joint pose used
    by HumanoidWalkEnv.

    Parameters
    ----------
    keypoints_all : np.ndarray

        Either:

            (num_people, 25, 3)

        or:

            (25, 3)

        Each BODY_25 point contains:

            [x, y, confidence]


    selected_index : int
        Index of the selected person.


    verbose : bool
        Print conversion information.


    Returns
    -------
    theta_init : np.ndarray, shape (8,)

        Final simulator-compatible joint pose:

        [
            right_hip,
            right_knee,
            left_hip,
            left_knee,
            left_shoulder,
            left_elbow,
            right_shoulder,
            right_elbow
        ]


    metadata : dict
        Additional information including:

        - selected person index
        - actuator order
        - raw segment orientations
        - unclipped humanoid angles
        - final joint angles
        - joint limits
        - clipping status
        - spine orientation
    """

    keypoints_all = np.asarray(
        keypoints_all
    )

    # ---------------------------------------------------------------------
    # Support a single BODY_25 skeleton
    # ---------------------------------------------------------------------

    if keypoints_all.ndim == 2:

        keypoints_all = np.expand_dims(
            keypoints_all,
            axis=0,
        )

    # ---------------------------------------------------------------------
    # Validate input
    # ---------------------------------------------------------------------

    if keypoints_all.ndim != 3:

        raise ValueError(
            "keypoints_all must have shape "
            "(num_people, 25, 3) or (25, 3)"
        )

    if keypoints_all.shape[1] < 25:

        raise ValueError(
            "Expected BODY_25 keypoints, "
            f"received shape {keypoints_all.shape}"
        )

    if (
        selected_index < 0
        or selected_index
        >= keypoints_all.shape[0]
    ):

        raise IndexError(
            f"selected_index={selected_index} "
            "is invalid for "
            f"{keypoints_all.shape[0]} detected people"
        )

    skeleton = keypoints_all[
        selected_index
    ]

    segment_angles = []
    angle_details = []

    if verbose:

        print(
            "=" * 100
        )

        print(
            "CONVERTING BODY_25 SKELETON TO "
            "8-JOINT HUMANOID POSE"
        )

        print(
            f"Selected Person: "
            f"{selected_index + 1}"
        )

        print(
            "=" * 100
        )

        print()

    # ---------------------------------------------------------------------
    # Calculate the 8 BODY_25 segment orientations
    # ---------------------------------------------------------------------

    for actuator_index, (
        actuator_name,
        joint_a,
        joint_b,
    ) in enumerate(
        ACTUATOR_PAIRS
    ):

        angle, details = _segment_angle(
            skeleton,
            joint_a,
            joint_b,
        )

        segment_angles.append(
            angle
        )

        if details is None:

            details = {
                "actuator_index": actuator_index,

                "actuator": actuator_name,

                "joint_a": BODY_25_NAMES[
                    joint_a
                ],

                "joint_b": BODY_25_NAMES[
                    joint_b
                ],

                "angle_rad": 0.0,

                "angle_deg": 0.0,

                "status": "invalid",
            }

        else:

            details[
                "actuator_index"
            ] = actuator_index

            details[
                "actuator"
            ] = actuator_name

        angle_details.append(
            details
        )

        if verbose:

            print(
                f"{actuator_index}: "
                f"{actuator_name:<16} "
                f"{details['joint_a']} "
                "-> "
                f"{details['joint_b']} "
                f"segment={angle:+.4f} rad "
                f"({np.degrees(angle):+.2f} deg)"
            )

    # ---------------------------------------------------------------------
    # Convert absolute segment directions into relative humanoid angles
    # ---------------------------------------------------------------------

    segment_orientations = np.asarray(
        segment_angles,
        dtype=np.float32,
    )

    unclipped_theta = (
        _segments_to_humanoid_pose(
            segment_orientations
        )
    )

    # ---------------------------------------------------------------------
    # Enforce the same joint limits as env/humanoid.urdf
    # ---------------------------------------------------------------------

    lower_limits = (
        HUMANOID_JOINT_LIMITS[
            :,
            0,
        ]
    )

    upper_limits = (
        HUMANOID_JOINT_LIMITS[
            :,
            1,
        ]
    )

    theta_init = np.clip(
        unclipped_theta,
        lower_limits,
        upper_limits,
    ).astype(
        np.float32
    )

    pose_was_clipped = bool(
        not np.allclose(
            unclipped_theta,
            theta_init,
            atol=1e-7,
        )
    )

    # ---------------------------------------------------------------------
    # Spine orientation
    #
    # Stored for metadata only.
    # ---------------------------------------------------------------------

    (
        spine_angle,
        spine_details,
    ) = _segment_angle(
        skeleton,
        SPINE_PAIR[0],
        SPINE_PAIR[1],
    )

    # ---------------------------------------------------------------------
    # Metadata
    # ---------------------------------------------------------------------

    metadata = {
        "selected_person_index": int(
            selected_index
        ),

        "num_joints": int(
            len(theta_init)
        ),

        "actuator_order": [
            item[0]
            for item in ACTUATOR_PAIRS
        ],

        "segment_orientations_rad": [
            float(x)
            for x in segment_orientations
        ],

        "unclipped_pose_rad": [
            float(x)
            for x in unclipped_theta
        ],

        "final_pose_rad": [
            float(x)
            for x in theta_init
        ],

        "pose_was_clipped": (
            pose_was_clipped
        ),

        "joint_limits_rad": {
            ACTUATOR_PAIRS[
                i
            ][0]: [
                float(
                    HUMANOID_JOINT_LIMITS[
                        i,
                        0,
                    ]
                ),

                float(
                    HUMANOID_JOINT_LIMITS[
                        i,
                        1,
                    ]
                ),
            ]

            for i in range(
                len(
                    ACTUATOR_PAIRS
                )
            )
        },

        "angle_details": (
            angle_details
        ),

        "spine_angle_rad": float(
            spine_angle
        ),

        "spine_angle_deg": float(
            np.degrees(
                spine_angle
            )
        ),

        "spine_details": (
            spine_details
        ),
    }

    # ---------------------------------------------------------------------
    # Verbose output
    # ---------------------------------------------------------------------

    if verbose:

        print()
        print(
            "-" * 100
        )

        print(
            "FINAL HUMANOID JOINT ANGLES"
        )

        print(
            "-" * 100
        )

        actuator_order = metadata[
            "actuator_order"
        ]

        for index, (
            actuator_name,
            angle,
        ) in enumerate(
            zip(
                actuator_order,
                theta_init,
            )
        ):

            lower = (
                HUMANOID_JOINT_LIMITS[
                    index,
                    0,
                ]
            )

            upper = (
                HUMANOID_JOINT_LIMITS[
                    index,
                    1,
                ]
            )

            original_angle = (
                unclipped_theta[
                    index
                ]
            )

            was_joint_clipped = (
                not np.isclose(
                    original_angle,
                    angle,
                    atol=1e-7,
                )
            )

            status = (
                " CLIPPED"
                if was_joint_clipped
                else ""
            )

            print(
                f"{index}: "
                f"{actuator_name:<16} "
                f"{angle:+.4f} rad "
                f"({np.degrees(angle):+.2f} deg) "
                f"limit=[{lower:+.2f}, {upper:+.2f}]"
                f"{status}"
            )

        print()

        print(
            "Unclipped pose:",
            unclipped_theta,
        )

        print(
            "Final pose:",
            theta_init,
        )

        print(
            "Pose shape:",
            theta_init.shape,
        )

        print(
            "Pose required clipping:",
            pose_was_clipped,
        )

        print(
            "Spine orientation "
            "(metadata only):",
            f"{spine_angle:+.4f} rad "
            f"({np.degrees(spine_angle):+.2f} deg)",
        )

        print(
            "=" * 100
        )

    return (
        theta_init,
        metadata,
    )


# -------------------------------------------------------------------------
# Direct module test
# -------------------------------------------------------------------------

if __name__ == "__main__":

    print(
        "angles.py module ready"
    )

    print(
        "Actuator order:"
    )

    for index, (
        name,
        _,
        _,
    ) in enumerate(
        ACTUATOR_PAIRS
    ):

        lower = (
            HUMANOID_JOINT_LIMITS[
                index,
                0,
            ]
        )

        upper = (
            HUMANOID_JOINT_LIMITS[
                index,
                1,
            ]
        )

        print(
            f"  {index}: "
            f"{name:<16} "
            f"[{lower:+.2f}, {upper:+.2f}] rad"
        )