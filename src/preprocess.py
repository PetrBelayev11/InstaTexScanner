import cv2
import numpy as np
from pathlib import Path

def preprocess_image(path: Path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (256, 64))
    _, binarized = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binarized / 255.0

if __name__ == "__main__":
    p = Path("data/samples/sample_0.png")
    processed = preprocess_image(p)
    cv2.imwrite("data/samples/sample_0_preprocessed.png", (processed * 255).astype(np.uint8))
    print("Saved preprocessed example.")
