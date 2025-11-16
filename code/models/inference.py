from .latex_converter import LatexConverter

converter = LatexConverter()

def run_inference(image_path: str):
    return converter.convert(image_path)
