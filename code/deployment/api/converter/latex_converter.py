from pathlib import Path
import os
import shutil
from .cnn_detector import TinyEMNISTDetector
from .handwriting_classifier import is_handwritten_heuristic
from .trocr_ocr import TrocrEngine, PRINTED_MODEL, HANDWRITTEN_MODEL
from .document_segmenter import DocumentSegmenter, DocumentSegment


class LatexConverter:

    def __init__(self, enable_segmentation: bool = True):
        """
        Initialize LatexConverter.
        
        Args:
            enable_segmentation: If True, segment document into text and images
        """
        self.cnn = TinyEMNISTDetector()
        self.ocr_printed = TrocrEngine(PRINTED_MODEL)
        self.ocr_hand = TrocrEngine(HANDWRITTEN_MODEL)
        self.segmenter = DocumentSegmenter() if enable_segmentation else None

    def classify_handwriting(self, image_path: str) -> bool:
        try:
            return self.cnn.is_handwritten(image_path)
        except:
            return is_handwritten_heuristic(image_path)

    def process_text_region(self, image_path: str, bbox=None) -> str:
        """
        Process a text region with OCR.
        
        Args:
            image_path: Path to image or text region
            bbox: Optional bounding box for region extraction
            
        Returns:
            Extracted text
        """
        # If bbox is provided, extract region first
        if bbox is not None:
            import cv2
            import tempfile
            img = cv2.imread(image_path)
            x, y, width, height = bbox
            region = img[y:y+height, x:x+width]
            # Save temporary region
            temp_dir = tempfile.gettempdir()
            temp_path = os.path.join(temp_dir, f"text_region_{id(bbox)}.png")
            cv2.imwrite(temp_path, region)
            image_path = temp_path
        
        is_hand = self.classify_handwriting(image_path)
        
        if not is_hand:
            text = self.ocr_printed.run(image_path)
        else:
            text = self.ocr_hand.run(image_path)
        
        # Cleanup temp file if created
        if bbox is not None and os.path.exists(image_path):
            try:
                os.remove(image_path)
            except:
                pass
        
        return text

    def convert(self, image_path: str, out_dir="results", segment_document: bool = None):
        """
        Convert document image to LaTeX.
        
        Args:
            image_path: Path to input image
            out_dir: Output directory for LaTeX and images
            segment_document: Override segmentation setting (None uses init setting)
            
        Returns:
            Dictionary with conversion results
        """
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        
        # Determine if segmentation should be used
        use_segmentation = segment_document if segment_document is not None else (self.segmenter is not None)
        
        if use_segmentation:
            return self._convert_with_segmentation(image_path, out_dir)
        else:
            return self._convert_simple(image_path, out_dir)

    def _convert_simple(self, image_path: str, out_dir: str) -> dict:
        """Simple conversion without segmentation (backward compatibility)."""
        is_hand = self.classify_handwriting(image_path)

        if not is_hand:
            text = self.ocr_printed.run(image_path)
        else:
            text = self.ocr_hand.run(image_path)

        latex = f"""
\\documentclass{{article}}
\\usepackage[utf8]{{inputenc}}
\\usepackage{{amsmath}}
\\usepackage{{graphicx}}
\\begin{{document}}
{text}
\\end{{document}}
"""

        out_name = f"{Path(image_path).stem}_output.tex"
        out_path = Path(out_dir) / out_name
        out_path.write_text(latex)

        return {
            "type": "handwritten" if is_hand else "printed",
            "text": text,
            "latex_file": str(out_path),
            "images": []
        }

    def _convert_with_segmentation(self, image_path: str, out_dir: str) -> dict:
        """Convert with document segmentation."""
        # Create images subdirectory
        images_dir = Path(out_dir) / "images"
        images_dir.mkdir(exist_ok=True)
        
        # Segment document
        segments = self.segmenter.segment(image_path, str(images_dir))
        
        # Process segments
        latex_parts = []
        extracted_images = []
        all_text = []
        
        for segment in segments:
            if segment.type == "image":
                # Handle image segment
                image_filename = Path(segment.image_path).name
                # Copy image to output directory if not already there
                if not Path(segment.image_path).parent.samefile(images_dir):
                    dest_path = images_dir / image_filename
                    shutil.copy2(segment.image_path, dest_path)
                    segment.image_path = str(dest_path)
                
                # Add LaTeX include command
                # Use relative path from LaTeX file location
                rel_image_path = f"images/{image_filename}"
                latex_parts.append(f"\\includegraphics[width=\\textwidth]{{{rel_image_path}}}")
                extracted_images.append(rel_image_path)
            
            elif segment.type == "text":
                # Process text region
                text = self.process_text_region(image_path, segment.bbox)
                if text.strip():
                    latex_parts.append(text)
                    all_text.append(text)
        
        # Combine all text for return value
        combined_text = "\n".join(all_text)
        
        # Determine document type (use majority or first segment)
        is_hand = self.classify_handwriting(image_path)
        
        # Generate LaTeX document
        latex_content = f"""
\\documentclass{{article}}
\\usepackage[utf8]{{inputenc}}
\\usepackage{{amsmath}}
\\usepackage{{graphicx}}
\\begin{{document}}

{chr(10).join(latex_parts)}

\\end{{document}}
"""
        
        out_name = f"{Path(image_path).stem}_output.tex"
        out_path = Path(out_dir) / out_name
        out_path.write_text(latex_content)

        return {
            "type": "handwritten" if is_hand else "printed",
            "text": combined_text,
            "latex_file": str(out_path),
            "images": extracted_images,
            "segments_count": len(segments)
        }
