import cv2
import numpy as np
import os
from pathlib import Path

def load_image(image_path: str):
    """
    Load an image from file using OpenCV and keep BGR format (original colors).
    
    Args:
        image_path: Path to image file
    
    Returns:
        img_bgr: BGR image as numpy array (preserves original colors)
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Failed to load image: {image_path}")
    
    return img_bgr


def display_image(img_bgr, title="Image"):
    """
    Display BGR image in window.
    
    Args:
        img_bgr: BGR image array
        title: Window title
    """
    cv2.imshow(title, img_bgr)
    print(f"Press any key to close the image window...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def save_image(img_bgr, output_path: str):
    """
    Save BGR image to file (preserves original colors).
    
    Args:
        img_bgr: BGR image array (as loaded by cv2.imread)
        output_path: Path to save image
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # img_bgr is already in BGR format, save directly without conversion
    cv2.imwrite(output_path, img_bgr)


def resize_image(img_bgr, max_width=1920, max_height=1080):
    """
    Resize image to fit max dimensions while preserving aspect ratio.
    
    Args:
        img_bgr: BGR image array
        max_width: Maximum width
        max_height: Maximum height
    
    Returns:
        resized_img: Resized BGR image
    """
    h, w = img_bgr.shape[:2]
    scale = min(max_width / w, max_height / h, 1.0)
    
    if scale < 1.0:
        new_w = int(w * scale)
        new_h = int(h * scale)
        resized = cv2.resize(img_bgr, (new_w, new_h))
        return resized
    
    return img_bgr


def get_image_info(image_path: str):
    """
    Get image information without loading full image.
    
    Args:
        image_path: Path to image file
    
    Returns:
        info: Dictionary with image info
    """
    if not os.path.exists(image_path):
        return None
    
    file_size = os.path.getsize(image_path) / (1024 * 1024)  # MB
    file_name = os.path.basename(image_path)
    
    return {
        'file_name': file_name,
        'file_path': image_path,
        'file_size_mb': file_size,
        'exists': True
    }


if __name__ == "__main__":
    print("OK image.py module ready")