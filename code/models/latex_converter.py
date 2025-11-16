from pathlib import Path
from .cnn_detector import TinyEMNISTDetector
from .handwriting_classifier import is_handwritten_heuristic
from .trocr_ocr import TrocrEngine, PRINTED_MODEL, HANDWRITTEN_MODEL


class LatexConverter:

    def __init__(self):
        self.cnn = TinyEMNISTDetector()
        self.ocr_printed = TrocrEngine(PRINTED_MODEL)
        self.ocr_hand = TrocrEngine(HANDWRITTEN_MODEL)

    def classify_handwriting(self, image_path: str) -> bool:
        try:
            return self.cnn.is_handwritten(image_path)
        except:
            return is_handwritten_heuristic(image_path)

    def convert(self, image_path: str, out_dir="results"):
        Path(out_dir).mkdir(exist_ok=True)

        is_hand = self.classify_handwriting(image_path)

        if not is_hand:
            text = self.ocr_printed.run(image_path)
        else:
            text = self.ocr_hand.run(image_path)

        latex = f"""
\\documentclass{{article}}
\\usepackage[utf8]{{inputenc}}
\\usepackage{{amsmath}}
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
            "latex_file": str(out_path)
        }
