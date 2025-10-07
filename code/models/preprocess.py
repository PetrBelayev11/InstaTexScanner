"""
preprocess.py

Simple preprocessing utilities for document images:
- convert to grayscale
- resize (optional)
- binarize (global threshold)
- save output image

Usage:
    python preprocess.py sample_input.png sample_output.png
"""
from PIL import Image
import sys

def binarize_image(in_path, out_path, threshold=180):
    im = Image.open(in_path).convert("L")  # grayscale
    # simple global threshold
    bw = im.point(lambda p: 255 if p > threshold else 0)
    bw.save(out_path)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python preprocess.py input.png output.png")
    else:
        binarize_image(sys.argv[1], sys.argv[2])
