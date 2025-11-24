import cv2
import numpy as np

def is_handwritten_heuristic(image_path: str) -> bool:
    img = cv2.imread(image_path, 0)
    _, th = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) < 5:
        return False

    areas = np.array([cv2.contourArea(c) for c in contours if cv2.contourArea(c) > 0])
    if len(areas) == 0:
        return False

    std = areas.std()
    small_ratio = (areas < np.percentile(areas, 40)).sum() / len(areas)

    return (std > 400 and small_ratio > 0.4)
