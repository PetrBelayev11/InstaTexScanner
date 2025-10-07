"""
data_and_model_demo.py

Demonstration script for InstaTexScanner project:
- Loads EMNIST handwritten dataset via torchvision (if available)
- Generates synthetic printed-text and handwritten-like images
- Runs simple preprocessing (grayscale, resize, binarize)
- Defines a tiny CNN and runs one training epoch on a small subset (optional)

"""

import os, math, random
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageOps
import numpy as np

OUT_DIR = Path("../../data")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_MODEL_DIR = Path("../../models")
OUT_MODEL_DIR.mkdir(parents=True, exist_ok=True)
SAMPLES_DIR = OUT_DIR / "samples"
SAMPLES_DIR.mkdir(exist_ok=True)

# --------------------- utils ---------------------
def ensure_font(preferred=("DejaVuSans.ttf","Arial.ttf")):
    # Try to find a font on system; fallback to default PIL font
    from PIL import ImageFont
    for name in preferred:
        try:
            return ImageFont.truetype(name, 32)
        except:
            continue
    return ImageFont.load_default()

def make_printed_text_image(text, size=(800,200), font=None):
    img = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(img)
    if font is None:
        font = ensure_font()
    # wrap text so it fits width
    lines = []
    maxw = size[0] - 40
    words = text.split()
    cur = ""
    for w in words:
        # compute text width depending on Pillow version
        candidate = (cur + " " + w).strip()
        try:
            # Pillow ≥10.0 (textbbox exists)
            bbox = draw.textbbox((0,0), candidate, font=font)
            text_width = bbox[2] - bbox[0]
        except AttributeError:
            # older Pillow
            text_width = draw.textsize(candidate, font=font)[0]
        if text_width <= maxw:
            cur = candidate
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    y = 20
    for line in lines:
        draw.text((20, y), line, fill="black", font=font)
        try:
            h = draw.textbbox((0,0), line, font=font)[3]
        except AttributeError:
            h = font.getsize(line)[1]
        y += h + 6
    return img


def make_handwritten_like_image(text, size=(800,200), font=None):
    # create printed text then apply distortions to mimic handwriting
    img = make_printed_text_image(text, size=size, font=font)
    img = img.convert("L")
    # apply slight blur and elastic-like distortion by offsetting strips
    img = img.filter(ImageFilter.GaussianBlur(radius=0.6))
    arr = np.array(img)
    h, w = arr.shape
    out = np.zeros_like(arr)
    for i in range(0, h, 4):
        shift = int(4 * math.sin(i / 15.0) + random.randint(-1,1))
        out[i:i+4, :] = np.roll(arr[i:i+4, :], shift, axis=1)
    out = Image.fromarray(out)
    # add ink variation
    out = ImageOps.autocontrast(out)
    return out.convert("RGB")

def preprocess_image(img, target_size=(256,256), bin_thresh=180):
    im = img.convert("L").resize(target_size, Image.BILINEAR)
    # binarize
    bw = im.point(lambda p: 255 if p > bin_thresh else 0)
    return im, bw

# --------------------- dataset loading (EMNIST) ---------------------
have_torch = False
try:
    import torch, torchvision
    from torchvision import transforms, datasets
    have_torch = True
except Exception as e:
    print("PyTorch/torchvision not available or failed to import:", e)
    have_torch = False

emnist_subset_info = None
if have_torch:
    try:
        # EMNIST has splits: 'byclass','bymerge','balanced','letters','digits','mnist'
        emnist_root = OUT_DIR / "emnist_data"
        transform = transforms.Compose([transforms.ToTensor()])
        print("Attempting to download EMNIST (this may take a while)...")
        emnist_train = datasets.EMNIST(root=str(emnist_root), split='letters', train=True, download=True, transform=transform)
        emnist_test = datasets.EMNIST(root=str(emnist_root), split='letters', train=False, download=True, transform=transform)
        emnist_subset_info = {
            "train_len": len(emnist_train),
            "test_len": len(emnist_test)
        }
    except Exception as e:
        print("Failed to download or load EMNIST:", e)
        emnist_train = None
        emnist_test = None
        have_torch = False

# --------------------- synthetic data generation ---------------------
SAMPLE_TEXTS = [
    "InstaTexScanner sample: E = mc^2 and \\int_0^1 x^2 dx = 1/3",
    "This is a printed text sample with numbers 123456 and punctuation.",
    "Handwriting-like sample: find f(x) = x^2 + 2x + 1",
    "Figure: simple diagram placeholder"
]

font = ensure_font(("DejaVuSans.ttf","LiberationSans-Regular.ttf","Arial.ttf"))
generated = []
for i, txt in enumerate(SAMPLE_TEXTS):
    p = SAMPLES_DIR / f"printed_{i}.png"
    img = make_printed_text_image(txt, size=(900,200), font=font)
    img.save(p)
    generated.append(p)
    p2 = SAMPLES_DIR / f"handwritten_{i}.png"
    img2 = make_handwritten_like_image(txt, size=(900,200), font=font)
    img2.save(p2)
    generated.append(p2)

# Preprocess the first few generated samples and save
for i, p in enumerate(generated[:4]):
    img = Image.open(p)
    im_resized, im_bw = preprocess_image(img, target_size=(512,256), bin_thresh=160)
    im_resized.save(SAMPLES_DIR / f"pre_{p.name}")
    im_bw.save(SAMPLES_DIR / f"bw_{p.name}")

# --------------------- small CNN model demo (optional, uses EMNIST if available) ---------------------
if have_torch and emnist_subset_info is not None:
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, Subset

    print("EMNIST sizes:", emnist_subset_info)
    # create tiny subset for quick demo training
    subset_indices = list(range(0, min(2000, len(emnist_train))))
    train_subset = Subset(emnist_train, subset_indices)
    train_loader = DataLoader(train_subset, batch_size=64, shuffle=True, num_workers=0)

    class TinyCNN(nn.Module):
        def __init__(self, num_classes=27):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Flatten(),
                nn.Linear(32*7*7, 128), nn.ReLU(),
                nn.Linear(128, num_classes)
            )
        def forward(self, x):
            return self.net(x)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyCNN(num_classes=27).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # simple 1-epoch training
    model.train()
    for epoch in range(1):
        running_loss = 0.0
        for i, (imgs, labels) in enumerate(train_loader):
            imgs = imgs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 20 == 0:
                print(f"Batch {i}, loss {loss.item():.4f}")
        print("Epoch done, avg loss:", running_loss / (i+1))

    
    # save tiny model
    torch.save(model.state_dict(), OUT_MODEL_DIR / "tiny_cnn_emnist.pth")
    print("Saved tiny model to", OUT_MODEL_DIR / "tiny_cnn_emnist.pth")

# --------------------- final report prints ---------------------
print("\\nGenerated sample images saved to:", SAMPLES_DIR)
if emnist_subset_info:
    print("EMNIST info:", emnist_subset_info)
else:
    print("EMNIST not available in this run; to use EMNIST, install torchvision and run the script locally.")
print("Script complete.")