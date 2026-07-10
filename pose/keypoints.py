import sys
import os
import cv2
import numpy as np

# Configure OpenPose paths
OPENPOSE_DIR = r"C:\openpose"

sys.path.append(os.path.join(OPENPOSE_DIR, "build", "python", "openpose", "Release"))
sys.path.append(os.path.join(OPENPOSE_DIR, "build", "python", "openpose"))

dll_paths = [
    os.path.join(OPENPOSE_DIR, "build", "x64", "Release"),
    os.path.join(OPENPOSE_DIR, "build", "bin"),
    os.path.join(OPENPOSE_DIR, "3rdparty", "caffe", "bin"),
]
os.environ["PATH"] += ";" + ";".join(dll_paths)

# Import OpenPose
try:
    import pyopenpose as op # type: ignore
    OPENPOSE_AVAILABLE = True
except Exception as e:
    OPENPOSE_AVAILABLE = False
    print("Failed to load OpenPose. Check DLL paths and Python version.")
    raise e

# BODY_25 keypoint names
BODY_25_NAMES = [
    "Nose", "Neck", "RShoulder", "RElbow", "RWrist",
    "LShoulder", "LElbow", "LWrist", "MidHip",
    "RHip", "RKnee", "RAnkle", "LHip", "LKnee",
    "LAnkle", "REye", "LEye", "REar", "LEar",
    "LBigToe", "LSmallToe", "LHeel",
    "RBigToe", "RSmallToe", "RHeel", "Background"
]

# Body skeleton connections (BODY_25 model)
# These are the limb connections for drawing skeleton properly
BODY_25_CONNECTIONS = [
    (0, 1),    # Nose to Neck
    (1, 2),    # Neck to RShoulder
    (2, 3),    # RShoulder to RElbow
    (3, 4),    # RElbow to RWrist
    (1, 5),    # Neck to LShoulder
    (5, 6),    # LShoulder to LElbow
    (6, 7),    # LElbow to LWrist
    (1, 8),    # Neck to MidHip
    (8, 9),    # MidHip to RHip
    (9, 10),   # RHip to RKnee
    (10, 11),  # RKnee to RAnkle
    (8, 12),   # MidHip to LHip
    (12, 13),  # LHip to LKnee
    (13, 14),  # LKnee to LAnkle
    (0, 15),   # Nose to REye
    (15, 17),  # REye to REar
    (0, 16),   # Nose to LEye
    (16, 18),  # LEye to LEar
    (11, 22),  # RAnkle to RHeel
    (22, 23),  # RHeel to RBigToe
    (11, 24),  # RAnkle to RSmallToe
    (14, 19),  # LAnkle to LHeel
    (19, 20),  # LHeel to LBigToe
    (14, 21),  # LAnkle to LSmallToe
]


def get_bounding_box(keypoints):
    """
    Calculate bounding box for a person.
    
    Args:
        keypoints: Array of shape (25, 3)
    
    Returns:
        bbox: Tuple (x_min, y_min, x_max, y_max)
        area: Bounding box area
        width: Width of bounding box
        height: Height of bounding box
    """
    valid_points = keypoints[keypoints[:, 2] > 0.1][:, :2]
    
    if valid_points.size == 0:
        return (0, 0, 0, 0), 0, 0, 0
    
    x_min = int(np.min(valid_points[:, 0]))
    y_min = int(np.min(valid_points[:, 1]))
    x_max = int(np.max(valid_points[:, 0]))
    y_max = int(np.max(valid_points[:, 1]))
    
    width = x_max - x_min
    height = y_max - y_min
    area = width * height
    
    return (x_min, y_min, x_max, y_max), area, width, height


def print_keypoints_detailed(keypoints, person_id):
    """Print detailed keypoint information."""
    
    print(f"\n{'='*100}")
    print(f"KEYPOINT DETAILS - PERSON {person_id + 1}")
    print(f"{'='*100}")
    print(f"{'#':<4} {'Joint Name':<15} {'X (pixels)':<15} {'Y (pixels)':<15} {'Confidence':<12} {'Status':<10}")
    print(f"{'-'*100}")
    
    for kpt_idx, (x, y, conf) in enumerate(keypoints[:25]):
        joint_name = BODY_25_NAMES[kpt_idx]
        
        if conf > 0.1:
            status = "OK"
            symbol = "OK"
        else:
            status = "INVALID"
            symbol = "XX"
        
        print(f"{symbol} {kpt_idx:<2}  {joint_name:<15} {x:<15.2f} {y:<15.2f} {conf:<12.4f} {status:<10}")
    
    print(f"{'-'*100}\n")


def draw_skeleton_on_image(img, keypoints, bbox, person_id, area, is_selected=False):
    """
    Draw skeleton and bounding box on image.
    Uses proper BODY_25 skeleton connections for better visualization.
    
    Args:
        img: Input image (BGR)
        keypoints: Keypoints for person (25, 3)
        bbox: Bounding box coordinates
        person_id: Person ID
        area: Bounding box area
        is_selected: Whether this is the selected person
    
    Returns:
        Modified image
    """
    
    # Draw bounding box
    if is_selected:
        thickness_box = 3
        box_color = (0, 255, 0)  # Green
    else:
        thickness_box = 2
        box_color = (0, 0, 255)  # Red
    
    cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), box_color, thickness_box)
    
    # Draw skeleton connections (limbs)
    for connection in BODY_25_CONNECTIONS:
        start_idx, end_idx = connection
        
        # Get keypoints
        pt1 = keypoints[start_idx]
        pt2 = keypoints[end_idx]
        
        # Check if both points are valid (confidence > 0.1)
        if pt1[2] > 0.1 and pt2[2] > 0.1:
            pt1_pos = (int(pt1[0]), int(pt1[1]))
            pt2_pos = (int(pt2[0]), int(pt2[1]))
            
            # Color for skeleton
            if is_selected:
                skeleton_color = (0, 255, 255)  # Yellow/Cyan for selected
                line_thickness = 2
            else:
                skeleton_color = (100, 100, 100)  # Gray for others
                line_thickness = 1
            
            # Draw line between keypoints
            cv2.line(img, pt1_pos, pt2_pos, skeleton_color, line_thickness)
    
    # Draw keypoints (circles)
    for kpt_idx, (x, y, conf) in enumerate(keypoints[:25]):
        if conf > 0.2:
            pos = (int(x), int(y))
            
            if is_selected:
                kpt_color = (0, 255, 255)  # Yellow
                radius = 5
                cv2.circle(img, pos, radius, kpt_color, -1)
                
                # Draw keypoint name for selected person
                if conf > 0.5:
                    cv2.putText(img, BODY_25_NAMES[kpt_idx],
                               (int(x) + 8, int(y) - 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 1)
            else:
                kpt_color = (150, 150, 150)  # Gray
                radius = 3
                cv2.circle(img, pos, radius, kpt_color, -1)
    
    # Draw label
    if is_selected:
        label_text = f"SELECTED: Person {person_id + 1} - Area: {area:.0f}px2"
        label_color = (0, 255, 0)  # Green
    else:
        label_text = f"Person {person_id + 1} - Area: {area:.0f}px2"
        label_color = (0, 0, 255)  # Red
    
    # Draw text background
    (text_width, text_height), _ = cv2.getTextSize(
        label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    
    cv2.rectangle(img,
                 (bbox[0], bbox[1] - text_height - 8),
                 (bbox[0] + text_width + 8, bbox[1]),
                 label_color, -1)
    
    # Draw text
    cv2.putText(img, label_text, (bbox[0] + 4, bbox[1] - 4),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return img


def extract_keypoints(image_path: str, return_image=False, verbose=True):
    """
    Run OpenPose BODY_25 model to extract 25-keypoint skeletons for ALL people.
    
    Args:
        image_path: Path to image file
        return_image: If True, also return annotated image
        verbose: Print detailed information
    
    Returns:
        keypoints: np.ndarray of shape (num_people, 25, 3)
        output_image: Annotated image with all skeletons (if return_image=True)
        max_area_idx: Index of person with largest area
        people_data: List of dictionaries with info for each person
    """
    
    if not OPENPOSE_AVAILABLE:
        raise RuntimeError("OpenPose not available")
    
    if verbose:
        print(f"\n{'='*100}")
        print(f"EXTRACTING 25-KEYPOINT SKELETON FOR ALL PEOPLE")
        print(f"{'='*100}\n")
    
    # Load original image (preserve colors)
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot load image: {image_path}")
    
    # Keep copy of original for output
    output_img = img_bgr.copy()
    
    if verbose:
        print(f"Image loaded: {img_bgr.shape}\n")
    
    # Configure OpenPose
    params = dict(
        model_folder=os.path.join(OPENPOSE_DIR, "models"),
        model_pose="BODY_25"
    )
    
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()
    
    # Extract keypoints
    datum = op.Datum()
    datum.cvInputData = img_bgr
    
    datums = op.VectorDatum()
    datums.append(datum)
    opWrapper.emplaceAndPop(datums)
    
    if datum.poseKeypoints is None or datum.poseKeypoints.size == 0:
        raise ValueError("No people detected in image")
    
    keypoints = np.array(datum.poseKeypoints)
    num_people = keypoints.shape[0]
    
    if verbose:
        print(f"OK Detected {num_people} person(s)")
        print(f"OK Keypoints shape: {keypoints.shape}\n")
    
    # Print detailed keypoint info for each person
    if verbose:
        for person_idx in range(num_people):
            print_keypoints_detailed(keypoints[person_idx], person_idx)
    
    # Analyze each person
    print(f"{'='*100}")
    print(f"ANALYZING EACH DETECTED PERSON")
    print(f"{'='*100}\n")
    
    people_data = []
    
    for person_idx in range(num_people):
        person_kpts = keypoints[person_idx]
        bbox, area, width, height = get_bounding_box(person_kpts)
        valid_kpts = np.sum(person_kpts[:, 2] > 0.1)
        
        person_info = {
            'person_id': person_idx,
            'keypoints': person_kpts,
            'bbox': bbox,
            'area': area,
            'width': width,
            'height': height,
            'valid_keypoints': valid_kpts
        }
        
        people_data.append(person_info)
        
        print(f"PERSON {person_idx + 1}:")
        print(f"  Bounding Box: ({bbox[0]}, {bbox[1]}) to ({bbox[2]}, {bbox[3]})")
        print(f"  Width x Height: {width} x {height}")
        print(f"  Area: {area} pixels2")
        print(f"  Valid Keypoints: {valid_kpts}/25")
        print()
    
    # Compare areas and select largest
    print(f"{'='*100}")
    print(f"AREA COMPARISON & SELECTION")
    print(f"{'='*100}\n")
    
    print(f"Area Comparison:")
    areas = [p['area'] for p in people_data]
    max_area_idx = np.argmax(areas)
    
    for idx, p in enumerate(people_data):
        is_largest = "<- LARGEST" if idx == max_area_idx else ""
        print(f"  Person {idx + 1}: {p['area']:>8.0f} pixels2 {is_largest}")
    
    selected_person = people_data[max_area_idx]
    
    print(f"\n{'-'*100}")
    print(f"SELECTED: Person {selected_person['person_id'] + 1}")
    print(f"  Reason: Largest bounding box area ({selected_person['area']:.0f} pixels2)")
    print(f"{'-'*100}\n")
    
    # Draw visualization on original image
    if return_image:
        # Draw all people
        for idx, person_info in enumerate(people_data):
            is_selected = (idx == max_area_idx)
            
            output_img = draw_skeleton_on_image(
                output_img,
                person_info['keypoints'],
                person_info['bbox'],
                person_info['person_id'],
                person_info['area'],
                is_selected=is_selected
            )
        
        # Add summary text at top
        summary_text = f"Total People: {num_people}"
        cv2.putText(output_img, summary_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        selected_text = f"Selected: Person {selected_person['person_id'] + 1} (Largest)"
        cv2.putText(output_img, selected_text, (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        return keypoints, output_img, max_area_idx, people_data
    
    return keypoints, None, max_area_idx, people_data


if __name__ == "__main__":
    print("OK keypoints.py module ready - Use with test_pose.py")