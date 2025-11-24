"""
Image detection module for detecting images/figures in documents.
Uses OpenCV for contour detection and heuristics to identify image regions.
"""
import cv2
import numpy as np
from typing import List, Tuple
from pathlib import Path


class ImageDetector:
    """
    Detects image regions in documents using OpenCV contour detection.
    """
    
    def __init__(self, min_area: int = 5000, aspect_ratio_range: Tuple[float, float] = (0.3, 3.0)):
        """
        Initialize image detector.
        
        Args:
            min_area: Minimum area for a region to be considered an image
            aspect_ratio_range: Valid aspect ratio range (width/height)
        """
        self.min_area = min_area
        self.aspect_ratio_range = aspect_ratio_range
    
    def detect_images(self, image_path: str) -> List[Tuple[int, int, int, int]]:
        """
        Detect image regions in a document.
        
        Args:
            image_path: Path to input image
            
        Returns:
            List of bounding boxes as (x, y, width, height)
        """
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply adaptive thresholding to separate content from background
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        image_regions = []
        h, w = gray.shape
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_area:
                continue
            
            x, y, width, height = cv2.boundingRect(contour)
            
            # Filter by aspect ratio
            aspect_ratio = width / height if height > 0 else 0
            if not (self.aspect_ratio_range[0] <= aspect_ratio <= self.aspect_ratio_range[1]):
                continue
            
            # Filter regions that are too small relative to image size
            if width < w * 0.05 or height < h * 0.05:
                continue
            
            # Check if region has high variance (likely an image, not text)
            roi = gray[y:y+height, x:x+width]
            if roi.size > 0:
                variance = np.var(roi)
                # Images typically have higher variance than text regions
                if variance > 500:  # Threshold may need adjustment
                    image_regions.append((x, y, width, height))
        
        # Sort by position (top to bottom, left to right)
        image_regions.sort(key=lambda box: (box[1], box[0]))
        
        return image_regions
    
    def extract_image(self, image_path: str, bbox: Tuple[int, int, int, int], 
                     output_path: str = None) -> np.ndarray:
        """
        Extract image region from document.
        
        Args:
            image_path: Path to source image
            bbox: Bounding box (x, y, width, height)
            output_path: Optional path to save extracted image
            
        Returns:
            Extracted image as numpy array
        """
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        x, y, width, height = bbox
        extracted = img[y:y+height, x:x+width]
        
        if output_path:
            cv2.imwrite(output_path, extracted)
        
        return extracted
    
    def is_image_region(self, image_path: str, bbox: Tuple[int, int, int, int]) -> bool:
        """
        Heuristic check if a region is likely an image (not text).
        
        Args:
            image_path: Path to source image
            bbox: Bounding box (x, y, width, height)
            
        Returns:
            True if region is likely an image
        """
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return False
        
        x, y, width, height = bbox
        roi = img[y:y+height, x:x+width]
        
        if roi.size == 0:
            return False
        
        # Check variance (images have higher variance)
        variance = np.var(roi)
        
        # Check edge density (images have more edges)
        edges = cv2.Canny(roi, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # Heuristic: high variance and moderate edge density suggests image
        return variance > 1000 and edge_density > 0.1

