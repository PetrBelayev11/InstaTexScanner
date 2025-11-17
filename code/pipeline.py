import torch
from pathlib import Path
import subprocess

from code.models.preprocess import preprocess_for_model
from code.models.handwriting_classifier import HandwritingClassifier
from code.models.trocr_ocr import run_trocr
from code.models.model_im2markup import Im2Latex
from code.models.inference import greedy_decode
from code.models.latex_converter import text_to_latex
from code.vocab import build_vocab

device = "cuda" if torch.cuda.is_available() else "cpu"


def load_handwriting_classifier():
    model_path = Path("models/handwriting_detector.pth")
    if not model_path.exists():
        raise RuntimeError("Missing model: models/handwriting_detector.pth")

    model = HandwritingClassifier()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device).eval()
    return model


def load_im2latex_model(vocab):
    model_path = Path("models/im2latex_epoch5.pth")
    if not model_path.exists():
        raise RuntimeError("Missing handwritten formula model: models/im2latex_epoch5.pth")

    model = Im2Latex(len(vocab)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def classify(img_tensor, classifier):
    with torch.no_grad():
        out = classifier(img_tensor)
    pred = torch.argmax(out, dim=1).item()
    return "handwritten" if pred == 1 else "printed"


def image_to_latex(image_path):
    vocab = build_vocab()

    processed = preprocess_for_model(image_path)
    tensor = torch.tensor(processed).unsqueeze(0).unsqueeze(0).float().to(device)

    classifier = load_handwriting_classifier()
    text_type = classify(tensor, classifier)
    print(f"[Classifier] Detected: {text_type}")

    if text_type == "printed":
        text = run_trocr(image_path)
        latex = text_to_latex(text)
        return latex

    im2latex_model = load_im2latex_model(vocab)
    token_ids = greedy_decode(im2latex_model, tensor, vocab)

    inv = {i: c for c, i in vocab.items()}
    latex = "".join(inv.get(i, "") for i in token_ids)
    return latex


def save_latex(latex, out="output.tex"):
    content = f"""
\\documentclass{{article}}
\\usepackage{{amsmath}}
\\usepackage{{amssymb}}
\\usepackage{{amsfonts}}
\\begin{{document}}

{latex}

\\end{{document}}
"""
    with open(out, "w", encoding="utf-8") as f:
        f.write(content)
    return out


def compile_pdf(tex_file="output.tex"):
    subprocess.run(["pdflatex", tex_file], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return "output.pdf"


def run_pipeline(image_path):
    print("[Pipeline] Starting…")
    latex = image_to_latex(image_path)
    tex_file = save_latex(latex)
    pdf_file = compile_pdf(tex_file)
    print(f"[Pipeline] Done. PDF saved: {pdf_file}")
    return latex
