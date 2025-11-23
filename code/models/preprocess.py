import cv2
import numpy as np
from pathlib import Path

def resize_and_pad(img_gray, target_h=64, target_w=256):
    h, w = img_gray.shape[:2]
    scale = min(target_w / w, target_h / h)
    nw, nh = int(w * scale), int(h * scale)
    resized = cv2.resize(img_gray, (nw, nh), interpolation=cv2.INTER_AREA)
    # pad (top, bottom, left, right)
    top = (target_h - nh) // 2
    bottom = target_h - nh - top
    left = (target_w - nw) // 2
    right = target_w - nw - left
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=255)
    return padded

def preprocess_save(in_path: str, out_path: str, target_h=64, target_w=256):
    p = Path(in_path)
    img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    img = resize_and_pad(img, target_h, target_w)
    _, th = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cv2.imwrite(str(out_path), th)

def preprocess_for_model(image_path: str, target_h=64, target_w=256):
    """
    Preprocess image for model input.
    Returns normalized numpy array ready for model.
    """
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    img = resize_and_pad(img, target_h, target_w)
    _, th = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Normalize to [0, 1] and invert (black=1, white=0)
    th = (255 - th) / 255.0
    return th

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python preprocess.py input.png output.png")
    else:
        preprocess_save(sys.argv[1], sys.argv[2])
