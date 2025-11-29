import cv2
import numpy as np
from pathlib import Path

def detect_content_bounding_box(img, background_threshold=240, padding=10):
    """
    Detect the bounding box of actual content (non-background) in the image.
    """
    # For transparent images, convert to binary mask
    if len(img.shape) == 3 and img.shape[2] == 4:  # RGBA
        alpha = img[:, :, 3]
        mask = alpha > 10  # Non-transparent areas
    else:
        # For grayscale, threshold to find content
        mask = img < background_threshold
    
    if not np.any(mask):
        # No content detected, return entire image
        return 0, 0, img.shape[1], img.shape[0]
    
    # Find bounding box of content
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    
    y1, y2 = np.where(rows)[0][[0, -1]]
    x1, x2 = np.where(cols)[0][[0, -1]]
    
    # Add padding
    y1 = max(0, y1 - padding)
    y2 = min(img.shape[0], y2 + padding)
    x1 = max(0, x1 - padding)
    x2 = min(img.shape[1], x2 + padding)
    
    return x1, y1, x2, y2

def smart_crop_and_resize(img_gray, target_h=128, target_w=512, min_aspect_ratio=0.2, max_aspect_ratio=5.0):
    """
    Smart cropping and resizing that preserves content readability.
    """
    # Step 1: Detect and crop to content
    x1, y1, x2, y2 = detect_content_bounding_box(img_gray)
    cropped = img_gray[y1:y2, x1:x2]
    
    if cropped.size == 0:
        # Fallback if cropping failed
        cropped = img_gray
    
    # Step 2: Calculate resize scale while maintaining aspect ratio
    h, w = cropped.shape
    current_aspect = w / h
    
    # Constrain aspect ratio to reasonable bounds
    constrained_aspect = max(min_aspect_ratio, min(max_aspect_ratio, current_aspect))
    
    # Calculate dimensions that fit within target while preserving constrained aspect
    if constrained_aspect > (target_w / target_h):
        # Width-limited
        new_w = target_w
        new_h = int(target_w / constrained_aspect)
    else:
        # Height-limited
        new_h = target_h
        new_w = int(target_h * constrained_aspect)
    
    # Ensure minimum dimensions
    new_w = max(new_w, 32)
    new_h = max(new_h, 32)
    
    # Step 3: High-quality resize
    resized = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    
    # Step 4: Pad to target size (center the content)
    padded = np.ones((target_h, target_w), dtype=np.uint8) * 255
    
    # Calculate padding offsets (center the content)
    y_offset = (target_h - new_h) // 2
    x_offset = (target_w - new_w) // 2
    
    padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    
    return padded

def adaptive_binarize(img_gray, method='otsu', block_size=35, C=10):
    """
    Adaptive binarization that works better for mathematical formulas.
    """
    if method == 'otsu':
        # Global Otsu thresholding
        _, binary = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        # Adaptive thresholding - better for varying lighting
        binary = cv2.adaptiveThreshold(
            img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, block_size, C
        )
    
    return binary

def preprocess_for_model(image_path: str, target_h=128, target_w=512):
    """
    Improved preprocessing for mathematical formula images.
    Returns normalized numpy array ready for model.
    """
    # Step 1: Load image (handle both RGB and RGBA)
    img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Convert to grayscale if needed
    if len(img.shape) == 3:
        if img.shape[2] == 4:  # RGBA
            # Convert RGBA to grayscale, using alpha channel to handle transparency
            rgb = img[:, :, :3]
            alpha = img[:, :, 3]
            img_gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            # Set transparent areas to white
            img_gray[alpha < 10] = 255
        else:  # RGB
            img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        img_gray = img
    
    # Step 2: Smart crop and resize
    processed = smart_crop_and_resize(img_gray, target_h=target_h, target_w=target_w)
    
    # Step 3: Adaptive binarization
    # Try adaptive first, fall back to Otsu if it produces poor results
    binary_adaptive = adaptive_binarize(processed, method='adaptive')
    binary_otsu = adaptive_binarize(processed, method='otsu')
    
    # Choose the binarization that preserves more detail
    adaptive_non_white = np.sum(binary_adaptive < 128)
    otsu_non_white = np.sum(binary_otsu < 128)
    
    if adaptive_non_white > otsu_non_white * 0.3:  # Adaptive preserves reasonable detail
        binary = binary_adaptive
    else:
        binary = binary_otsu
    
    # Step 4: Noise removal (optional, but helpful for scanned images)
    kernel = np.ones((2, 2), np.uint8)
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    # Step 5: Normalize to [0, 1] and invert (black=1, white=0)
    normalized = (255 - cleaned) / 255.0
    
    return normalized

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python preprocess.py input.png output.png")
    else:
        # For command line usage, convert and save the processed image
        processed = preprocess_for_model(sys.argv[1])
        output_img = (processed * 255).astype(np.uint8)
        cv2.imwrite(str(sys.argv[2]), output_img)