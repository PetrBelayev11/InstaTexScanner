"""Test script for checking training setup."""
import json
import sys
import importlib.util
from pathlib import Path

print("Checking training setup...")

# Check data
json_path = Path("datasets/im2latex_prepared.json")
print(f"1. Checking JSON file: {json_path.exists()}")
if json_path.exists():
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"   - Keys: {list(data.keys())}")
    print(f"   - Train samples: {len(data.get('train', []))}")
    print(f"   - Val samples: {len(data.get('val', []))}")

# Check imports
print("\n2. Checking imports...")
try:
    spec = importlib.util.spec_from_file_location("model_im2markup", "code/models/model_im2markup.py")
    model_im2markup = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_im2markup)
    Im2Latex = model_im2markup.Im2Latex
    print("   ✓ model_im2markup imported")
except Exception as e:
    print(f"   ✗ Import error model_im2markup: {e}")
    Im2Latex = None

try:
    spec = importlib.util.spec_from_file_location("preprocess", "code/models/preprocess.py")
    preprocess = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(preprocess)
    preprocess_for_model = preprocess.preprocess_for_model
    print("   ✓ preprocess imported")
except Exception as e:
    print(f"   ✗ Import error preprocess: {e}")
    preprocess_for_model = None

try:
    spec = importlib.util.spec_from_file_location("vocab", "code/vocab.py")
    vocab_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(vocab_module)
    build_vocab = vocab_module.build_vocab
    print("   ✓ vocab imported")
except Exception as e:
    print(f"   ✗ Import error vocab: {e}")
    build_vocab = None

# Check PyTorch
print("\n3. Checking PyTorch...")
try:
    import torch
    print(f"   - Version: {torch.__version__}")
    print(f"   - CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   - CUDA device: {torch.cuda.get_device_name(0)}")
except Exception as e:
    print(f"   ✗ Import error PyTorch: {e}")
    torch = None

# Check model creation
print("\n4. Checking model creation...")
if Im2Latex and torch:
    try:
        vocab = {'<SOS>': 0, '<EOS>': 1, '<PAD>': 2, '<UNK>': 3, 'a': 4, 'b': 5}
        model = Im2Latex(vocab_size=len(vocab))
        print(f"   ✓ Model created (parameters: {sum(p.numel() for p in model.parameters()):,})")
    except Exception as e:
        print(f"   ✗ Error creating model: {e}")
else:
    print("   ⚠ Skipped (required modules not loaded)")

# Check preprocessing
print("\n5. Checking preprocessing...")
if json_path.exists() and data.get('train') and preprocess_for_model:
    try:
        first_item = data['train'][0]
        img_path = first_item['img_path']
        if Path(img_path).exists():
            img = preprocess_for_model(img_path)
            print(f"   ✓ Image processed: shape {img.shape}")
        else:
            print(f"   ⚠ Image not found: {img_path}")
    except Exception as e:
        print(f"   ✗ Preprocessing error: {e}")
else:
    print("   ⚠ Skipped (required modules not loaded)")

print("\nCheck completed!")

