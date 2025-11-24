"""
Document segmentation module for separating text and image regions.
"""
import cv2
import numpy as np
import os
from typing import List, Dict, Tuple
from pathlib import Path
from .image_detector import ImageDetector


class DocumentSegment:
    """Represents a segment of a document."""
    
    def __init__(self, bbox: Tuple[int, int, int, int], segment_type: str, 
                 content: str = None, image_path: str = None):
        """
        Initialize document segment.
        
        Args:
            bbox: Bounding box (x, y, width, height)
            segment_type: 'text' or 'image'
            content: Text content (for text segments)
            image_path: Path to extracted image (for image segments)
        """
        self.bbox = bbox
        self.type = segment_type
        self.content = content
        self.image_path = image_path
        self.y_position = bbox[1]  # For sorting
    
    def __lt__(self, other):
        """Sort segments by vertical position."""
        return self.y_position < other.y_position


class DocumentSegmenter:
    """
    Segments a document into text and image regions.
    """
    
    def __init__(self, image_detector: ImageDetector = None):
        """
        Initialize document segmenter.
        
        Args:
            image_detector: ImageDetector instance (creates new one if None)
        """
        self.image_detector = image_detector or ImageDetector()
    
    def segment(self, image_path: str, images_dir: str = "images") -> List[DocumentSegment]:
        """
        Segment document into text and image regions.
        
        Args:
            image_path: Path to input document image
            images_dir: Directory to save extracted images
            
        Returns:
            List of DocumentSegment objects, sorted by vertical position
        """
        Path(images_dir).mkdir(parents=True, exist_ok=True)
        
        # Detect image regions
        image_bboxes = self.image_detector.detect_images(image_path)
        
        # Load image to get dimensions
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")
        h, w = img.shape[:2]
        
        # Create segments
        segments = []
        
        # Add image segments
        image_stem = Path(image_path).stem
        for i, bbox in enumerate(image_bboxes):
            x, y, width, height = bbox
            image_filename = f"{image_stem}_img_{i}.png"
            image_output_path = Path(images_dir) / image_filename
            
            # Extract and save image
            self.image_detector.extract_image(image_path, bbox, str(image_output_path))
            
            segment = DocumentSegment(
                bbox=bbox,
                segment_type="image",
                image_path=str(image_output_path)
            )
            segments.append(segment)
        
        # Create text regions (everything not covered by images)
        # Use a simpler approach: create horizontal strips between images
        if not image_bboxes:
            # No images detected, entire document is text
            segments.append(DocumentSegment(
                bbox=(0, 0, w, h),
                segment_type="text"
            ))
        else:
            # Sort images by vertical position
            sorted_images = sorted(image_bboxes, key=lambda b: b[1])
            current_y = 0
            
            for bbox in sorted_images:
                x, y, width, height = bbox
                
                # Text region before this image (if there's space)
                if y > current_y + 10:  # Minimum 10px gap
                    text_bbox = (0, current_y, w, y - current_y)
                    segments.append(DocumentSegment(
                        bbox=text_bbox,
                        segment_type="text"
                    ))
                
                current_y = max(current_y, y + height)
            
            # Text region after last image
            if current_y < h - 10:  # Minimum 10px at bottom
                text_bbox = (0, current_y, w, h - current_y)
                segments.append(DocumentSegment(
                    bbox=text_bbox,
                    segment_type="text"
                ))
        
        # Sort segments by vertical position
        segments.sort()
        
        return segments
    
    def extract_text_region(self, image_path: str, bbox: Tuple[int, int, int, int]) -> str:
        """
        Extract image region for text processing.
        
        Args:
            image_path: Path to source image
            bbox: Bounding box (x, y, width, height)
            
        Returns:
            Path to extracted text region image
        """
        import tempfile
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        x, y, width, height = bbox
        extracted = img[y:y+height, x:x+width]
        
        # Create temporary path
        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, f"text_region_{id(bbox)}.png")
        cv2.imwrite(temp_path, extracted)
        
        return temp_path

