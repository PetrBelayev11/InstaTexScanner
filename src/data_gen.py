import os
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import numpy as np
import random
from pathlib import Path

WINDOWS_FONT_PATHS = [
    "C:/Windows/Fonts/arial.ttf",
    "C:/Windows/Fonts/times.ttf",
    "C:/Windows/Fonts/cour.ttf",
    "C:/Windows/Fonts/calibri.ttf"
]

# Try to find available fonts
FONTS = []
for font_path in WINDOWS_FONT_PATHS:
    if os.path.exists(font_path):
        FONTS.append(font_path)

SAVE_DIR = Path("data/samples")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

def generate_sample(text: str, idx: int):
    img = Image.new("L", (320, 64), color=255)
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(random.choice(FONTS), 28)
    draw.text((10, 15), text, font=font, fill=0)

    img = img.rotate(random.uniform(-3, 3), expand=0, fillcolor=255)
    img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0, 1.2)))
    img.save(SAVE_DIR / f"sample_{idx}.png")

if __name__ == "__main__":
    samples = ["E=mc^2", "y = sin(x)", "∫ x^2 dx", "LaTeX Scanner"]
    for i, text in enumerate(samples):
        generate_sample(text, i)
    print(f"Generated {len(samples)} samples at {SAVE_DIR}")
