"""
COMPLETE BATCH PIPELINE: test_pose.py
For EACH image:
1. Detect ALL people
2. Show keypoint details (X, Y, Confidence)
3. Calculate area taken by EACH person
4. Compare areas
5. Process LARGEST area person (calculate angles)
6. Save training data for that person

Usage:
    python test_pose.py                    # Process all images
    python test_pose.py --single <image>   # Process single image
    python test_pose.py --help             # Show help
"""

import sys
import os
import numpy as np
import json
from pathlib import Path
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pose.image import load_image, save_image, resize_image
from pose.keypoints import extract_keypoints, BODY_25_NAMES
from pose.angles import skeleton_to_pose_vector


class CompleteBatchPipeline:
    """Complete batch processing pipeline."""
    
    def __init__(self, input_dir="data/input_images", output_dir="data/output_image"):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        
        # Create output directory if not exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Image extensions
        self.image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        
        # Statistics
        self.stats = {
            'total_images': 0,
            'processed': 0,
            'failed': 0,
            'total_people_detected': 0,
            'total_poses_extracted': 0,
            'failed_images': []
        }
    
    def find_all_images(self):
        """Find all images in input directory."""
        images = []
        
        for ext in self.image_extensions:
            images.extend(list(self.input_dir.glob(f"*{ext}")))
            images.extend(list(self.input_dir.glob(f"*{ext.upper()}")))
        
        # Remove duplicates and sort
        images = sorted(list(set(images)))
        
        return images
    
    def run(self):
        """Execute complete batch pipeline."""
        
        print("\n" + "╔" + "="*68 + "╗")
        print("║" + " "*12 + "COMPLETE BATCH PIPELINE" + " "*33 + "║")
        print("║" + " "*10 + "Detect All → Compare Areas → Process Largest" + " "*14 + "║")
        print("╚" + "="*68 + "╝\n")
        
        # Find images
        images = self.find_all_images()
        
        if not images:
            print(f"ERROR: No images found in: {self.input_dir}")
            print(f"\nPlease add images to: {self.input_dir}/ folder")
            print(f"Supported formats: jpg, jpeg, png, bmp, tiff\n")
            return False
        
        print(f"OK Found {len(images)} images in: {self.input_dir}\n")
        print(f"{'='*70}")
        print(f"OUTPUT DIRECTORY: {self.output_dir}")
        print(f"{'='*70}\n")
        
        self.stats['total_images'] = len(images)
        
        # Process each image
        for idx, image_path in enumerate(images, 1):
            self._process_single_image(image_path, idx, len(images))
        
        # Print summary
        self._print_summary()
        
        return True
    
    def _process_single_image(self, image_path, current_idx, total_images):
        """Process single image."""
        
        image_num = current_idx
        output_prefix = f"image{image_num}"
        
        print(f"\n{'╔' + '═'*68 + '╗'}")
        print(f"║ IMAGE {current_idx}/{total_images}: {image_path.name:<50} ║")
        print(f"{'╚' + '═'*68 + '╝'}")
        
        try:
            # ===== STAGE 1: Load Image =====
            print(f"\n  [1/4] Loading image...", end=" ")
            image = load_image(str(image_path))
            original_shape = image.shape
            image = resize_image(image, max_width=1920, max_height=1080)
            print(f"OK ({image.shape})")
            
            # ===== STAGE 2: Extract Keypoints (ALL PEOPLE) =====
            print(f"  [2/4] Extracting keypoints and analyzing...")
            keypoints, output_img, selected_idx, people_data = extract_keypoints(
                str(image_path),
                return_image=True,
                verbose=True
            )
            print(f"  [2/4] Keypoint extraction complete OK")
            
            if keypoints is None or keypoints.size == 0:
                print(f"\n  ERROR: No people detected")
                self.stats['failed'] += 1
                self.stats['failed_images'].append(image_path.name)
                return
            
            num_people = keypoints.shape[0]
            self.stats['total_people_detected'] += num_people
            
            # ===== STAGE 3: Display Area Comparison =====
            print(f"\n  [3/4] Area Analysis:")
            areas = [p['area'] for p in people_data]
            max_area = max(areas)
            
            for person_info in people_data:
                pid = person_info['person_id']
                area = person_info['area']
                is_selected = "<- LARGEST *" if pid == selected_idx else ""
                
                print(f"         Person {pid + 1}: {area:>8.0f} px2 {is_selected}")
            
            print(f"\n         Selected: Person {selected_idx + 1} ({max_area:.0f} px2)")
            
            # ===== STAGE 4: Calculate Angles (LARGEST PERSON) =====
            print(f"\n  [4/4] Calculating angles...")
            pose_vector, metadata = skeleton_to_pose_vector(
                keypoints,
                selected_index=selected_idx,
                verbose=True
            )
            print(f"  [4/4] Angle calculation complete OK")
            
            if pose_vector is None or len(pose_vector) == 0:
                print(f"\n  ERROR: Failed to calculate angles")
                self.stats['failed'] += 1
                self.stats['failed_images'].append(image_path.name)
                return
            
            # ===== SAVE OUTPUTS =====
            print(f"\n  Saving outputs...", end=" ")
            
            # Save keypoints visualization (with all people)
            keypoints_path = self.output_dir / f"{output_prefix}_keypoints.jpg"
            save_image(output_img, str(keypoints_path))
            
            # Save pose vector (NumPy)
            pose_npy_path = self.output_dir / f"{output_prefix}_pose_vector.npy"
            np.save(pose_npy_path, pose_vector)
            
            # Save pose vector (Text)
            pose_txt_path = self.output_dir / f"{output_prefix}_pose_vector.txt"
            try:
                with open(pose_txt_path, 'w', encoding='utf-8') as f:
                    f.write(f"Image: {image_path.name}\n")
                    f.write(f"Pose Vector (theta_init) - Person {selected_idx + 1}\n")
                    f.write(f"{'='*60}\n\n")
                    
                    f.write(f"People Detected: {num_people}\n")
                    f.write(f"Selected Person: {selected_idx + 1} (Area: {max_area:.0f} px2)\n\n")
                    
                    f.write(f"Joints: {len(pose_vector)}\n")
                    f.write(f"Shape: {pose_vector.shape}\n\n")
                    
                    f.write(f"Joint Angles:\n")
                    f.write(f"{'Index':<8} {'Radians':<15} {'Degrees':<15}\n")
                    f.write(f"{'─'*40}\n")
                    for i, angle in enumerate(pose_vector):
                        f.write(f"{i:<8} {angle:+.4f}      {np.degrees(angle):+8.2f}deg\n")
                    
                    f.write(f"\nStatistics:\n")
                    f.write(f"  Min: {np.min(pose_vector):+.4f} rad ({np.degrees(np.min(pose_vector)):+.2f}deg)\n")
                    f.write(f"  Max: {np.max(pose_vector):+.4f} rad ({np.degrees(np.max(pose_vector)):+.2f}deg)\n")
                    f.write(f"  Mean: {np.mean(pose_vector):+.4f} rad ({np.degrees(np.mean(pose_vector)):+.2f}deg)\n")
            except Exception as e:
                print(f"Warning: Could not save .txt file")
            
            # Save metadata (JSON)
            metadata_path = self.output_dir / f"{output_prefix}_metadata.json"
            
            # Area info for all people
            all_people_info = []
            for p in people_data:
                all_people_info.append({
                    'person_id': int(p['person_id']),
                    'area': float(p['area']),
                    'width': int(p['width']),
                    'height': int(p['height']),
                    'valid_keypoints': int(p['valid_keypoints']),
                    'is_selected': bool(p['person_id'] == selected_idx)
                })
            
            metadata_to_save = {
                'input_image': image_path.name,
                'image_shape': [int(x) for x in image.shape],
                'original_shape': [int(x) for x in original_shape],
                'num_people_detected': int(num_people),
                'all_people': all_people_info,
                'selected_person': int(selected_idx),
                'selected_person_area': float(max_area),
                'num_joints': int(len(pose_vector)),
                'pose_vector': [float(x) for x in pose_vector],
                'pose_vector_degrees': [float(np.degrees(x)) for x in pose_vector],
                'statistics': {
                    'min_rad': float(np.min(pose_vector)),
                    'max_rad': float(np.max(pose_vector)),
                    'mean_rad': float(np.mean(pose_vector)),
                    'std_rad': float(np.std(pose_vector)),
                    'min_deg': float(np.degrees(np.min(pose_vector))),
                    'max_deg': float(np.degrees(np.max(pose_vector))),
                    'mean_deg': float(np.degrees(np.mean(pose_vector)))
                },
                'angle_details': [
                    {
                        'joint_number': int(d.get('joint_number', 0)),
                        'limb': d.get('limb', 'N/A'),
                        'joint_a': d.get('joint_a', 'N/A'),
                        'joint_b': d.get('joint_b', 'N/A'),
                        'point_a': [float(x) for x in d.get('point_a', [0, 0, 0])] if 'point_a' in d else None,
                        'point_b': [float(x) for x in d.get('point_b', [0, 0, 0])] if 'point_b' in d else None,
                        'vector': [float(x) for x in d.get('vector', [0, 0])] if 'vector' in d else None,
                        'angle_rad': float(d.get('angle_rad', 0)),
                        'angle_deg': float(d.get('angle_deg', 0)),
                        'status': d.get('status', 'calculated')
                    }
                    for d in metadata['angle_details']
                ],
                'timestamp': datetime.now().isoformat()
            }
            
            try:
                with open(metadata_path, 'w', encoding='utf-8') as f:
                    json.dump(metadata_to_save, f, indent=2, ensure_ascii=False)
            except Exception as e:
                print(f"Warning: Could not save .json file")
            
            print(f"OK")
            
            # Print saved files
            print(f"\n  Output files:")
            print(f"    * {keypoints_path.name}")
            print(f"    * {pose_npy_path.name}")
            print(f"    * {pose_txt_path.name}")
            print(f"    * {metadata_path.name}")
            
            self.stats['processed'] += 1
            self.stats['total_poses_extracted'] += 1
        
        except Exception as e:
            print(f"ERROR")
            print(f"\n  ERROR: {str(e)}")
            self.stats['failed'] += 1
            self.stats['failed_images'].append(image_path.name)
    
    def _print_summary(self):
        """Print batch processing summary."""
        
        print(f"\n\n{'='*70}")
        print(f"BATCH PROCESSING COMPLETE")
        print(f"{'='*70}\n")
        
        print(f"Input directory: {self.input_dir}")
        print(f"Output directory: {self.output_dir}\n")
        
        print(f"Results:")
        print(f"  Total images: {self.stats['total_images']}")
        print(f"  Successfully processed: {self.stats['processed']} *")
        print(f"  Failed: {self.stats['failed']} x")
        print(f"  Total people detected: {self.stats['total_people_detected']}")
        print(f"  Total poses extracted: {self.stats['total_poses_extracted']}\n")
        
        if self.stats['failed'] > 0:
            print(f"Failed images:")
            for img in self.stats['failed_images']:
                print(f"  x {img}")
            print()
        
        if self.stats['processed'] > 0:
            print(f"* All outputs saved to: {self.output_dir}\n")
            print(f"Output for each image:")
            print(f"  * imageN_keypoints.jpg (All people + largest highlighted)")
            print(f"  * imageN_pose_vector.npy (Angles of largest person)")
            print(f"  * imageN_pose_vector.txt (Human readable)")
            print(f"  * imageN_metadata.json (Complete data for training)")
        
        print(f"\n{'='*70}\n")


def _build_training_dataset(poses_dir: str, out_path: str):
    """Aggregate all .npy pose vectors from a directory into a single dataset.

    - Recursively scans for files ending with "_pose_vector.npy" or any .npy files.
    - Ensures all vectors have identical length; skips incompatible ones with a warning.
    - Saves an .npz file with arrays: X (N, D) and filenames (N,).
    """
    poses_root = Path(poses_dir)
    if not poses_root.exists():
        print(f"\nERROR: Poses directory not found: {poses_dir}\n")
        return

    print("\n" + "="*70)
    print("BUILDING TRAINING DATASET FROM POSE VECTORS (.npy)")
    print("="*70)
    print(f"Input directory: {poses_root}")

    # Gather candidates
    candidates = []
    for p in poses_root.rglob("*.npy"):
        if p.name.endswith("_pose_vector.npy") or p.name.endswith(".npy"):
            candidates.append(p)

    candidates = sorted(candidates)

    if not candidates:
        print("\nNo .npy pose vectors found. Expected files like *_pose_vector.npy")
        return

    vectors = []
    names = []
    target_dim = None

    print(f"\nFound {len(candidates)} .npy files. Validating and loading...")

    for npy_path in candidates:
        try:
            vec = np.load(str(npy_path))
            # Flatten if needed
            if vec.ndim > 1:
                vec = vec.reshape(-1)

            if target_dim is None:
                target_dim = vec.shape[0]
            if vec.shape[0] != target_dim:
                print(f"  - SKIP (dim mismatch): {npy_path.name} has {vec.shape[0]} != {target_dim}")
                continue

            vectors.append(vec.astype(np.float32))
            names.append(npy_path.name)
        except Exception:
            print(f"  - SKIP (load error): {npy_path.name}")

    if not vectors:
        print("\nNo compatible pose vectors loaded.\n")
        return

    X = np.stack(vectors, axis=0)
    filenames = np.array(names)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(out_path), X=X, filenames=filenames)

    print("\nSaved dataset:")
    print(f"  Path: {out_path}")
    print(f"  Samples: {X.shape[0]}")
    print(f"  Vector dim: {X.shape[1]}")
    print("  Arrays: X (float32), filenames (str)")
    print("\nTip: Load with: data = np.load('<path>.npz'); X = data['X']")

def main():
    """Main entry point."""
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--help" or sys.argv[1] == "-h":
            print("""
COMPLETE BATCH PIPELINE: test_pose.py

Usage:
  python test_pose.py                    # Process all images
  python test_pose.py --single <path>    # Process single image
  python test_pose.py --dataset <poses_dir> [out_path]   # Build training dataset from .npy pose vectors
  python test_pose.py --help             # Show this help

Examples:
  python test_pose.py
  python test_pose.py --single data/input_images/photo.jpg
  python test_pose.py --dataset data/output_image data/datasets/pose_vectors.npz
            """)
            return
        
        elif sys.argv[1] == "--single" and len(sys.argv) > 2:
            image_path = sys.argv[2]
            if not os.path.exists(image_path):
                print(f"\nERROR: Image not found: {image_path}\n")
                return
            
            pipeline = CompleteBatchPipeline()
            pipeline._process_single_image(image_path, 1, 1)
            pipeline._print_summary()
            return
        
        elif sys.argv[1] == "--dataset" and len(sys.argv) > 2:
            poses_dir = sys.argv[2]
            out_path = sys.argv[3] if len(sys.argv) > 3 else os.path.join("data", "datasets", "pose_vectors.npz")
            _build_training_dataset(poses_dir, out_path)
            return
    
    # Default: Process all images in batch
    pipeline = CompleteBatchPipeline()
    pipeline.run()


if __name__ == "__main__":
    main()