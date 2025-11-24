"""
Script for converting images to LaTeX using trained model.
"""
import torch
import json
from pathlib import Path
import sys
import importlib.util

# Load model and vocabulary
def load_model_and_vocab(model_path, vocab_path):
    """Load trained model and vocabulary."""
    
    # Import model architecture
    spec = importlib.util.spec_from_file_location("model_im2markup", "code/models/model_im2markup.py")
    model_im2markup = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_im2markup)
    
    # Import preprocessing
    spec = importlib.util.spec_from_file_location("preprocess", "code/models/preprocess.py")
    preprocess = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(preprocess)
    
    # Load vocabulary
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab = json.load(f)
    
    vocab_size = len(vocab)
    
    # Create and load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_im2markup.Im2Latex(vocab_size=vocab_size).to(device)
    
    # Load weights (handle different formats)
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    
    return model, vocab, preprocess.preprocess_for_model, device

def greedy_decode(model, img_tensor, vocab, max_len=256):
    """Greedy decoder for inference."""
    model.eval()
    device = next(model.parameters()).device
    img_tensor = img_tensor.to(device)
    
    sos_token = vocab.get('<SOS>', 0)
    eos_token = vocab.get('<EOS>', 1)
    
    decoded = [sos_token]
    
    with torch.no_grad():
        enc_feat = model.encoder(img_tensor)
        hidden = None

        for _ in range(max_len):
            last_token = torch.tensor([[decoded[-1]]], dtype=torch.long, device=device)
            emb = model.decoder.embedding(last_token)

            query = model.decoder.query_proj(emb)
            attn_output, _ = model.decoder.attention(query, enc_feat, enc_feat)

            gru_input = torch.cat([emb, attn_output], dim=-1)
            out, hidden = model.decoder.gru(gru_input, hidden)
            logits = model.decoder.fc(out)

            next_token = logits.argmax(dim=-1).item()
            decoded.append(next_token)

            if next_token == eos_token:
                break

    # Remove SOS token and convert to text
    if decoded and decoded[0] == sos_token:
        decoded = decoded[1:]
    
    return decoded

def tokens_to_text(tokens, vocab):
    """Convert tokens to text string."""
    idx_to_char = {idx: char for char, idx in vocab.items()}
    eos_token = vocab.get('<EOS>', 1)
    pad_token = vocab.get('<PAD>', 2)
    
    text = []
    for token in tokens:
        if token == eos_token:
            break
        if token == pad_token:
            continue
        if token in idx_to_char:
            text.append(idx_to_char[token])
    return ''.join(text)

def image_to_latex(image_path, model, preprocess_fn, vocab):
    """Convert single image to LaTeX."""
    try:
        # Preprocess image
        img_array = preprocess_fn(image_path)
        img_tensor = torch.tensor(img_array, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # 1x1xHxW
        
        # Generate tokens
        tokens = greedy_decode(model, img_tensor, vocab)
        
        # Convert to text
        latex = tokens_to_text(tokens, vocab)
        
        return latex
        
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

def main():
    # Paths - UPDATE THESE!
    model_path = "models/im2latex_best.pth"  # Your model file
    vocab_path = "models/vocab.json"         # Vocabulary file
    image_path = "test_image.png"            # Your test image
    
    # Check files exist
    if not Path(model_path).exists():
        print(f"❌ Model file not found: {model_path}")
        return
        
    if not Path(vocab_path).exists():
        print(f"❌ Vocabulary file not found: {vocab_path}")
        return
        
    if not Path(image_path).exists():
        print(f"❌ Image file not found: {image_path}")
        return
    
    print("Loading model and vocabulary...")
    model, vocab, preprocess_fn, device = load_model_and_vocab(model_path, vocab_path)
    print(f"✓ Model loaded on {device}")
    print(f"✓ Vocabulary size: {len(vocab)}")
    
    print(f"\nConverting image: {image_path}")
    latex_result = image_to_latex(image_path, model, preprocess_fn, vocab)
    
    if latex_result:
        print("\n" + "="*50)
        print("✅ GENERATED LaTeX:")
        print("="*50)
        print(latex_result)
        print("="*50)
        
        # # Copy to clipboard (optional)
        # try:
        #     import pyperclip
        #     pyperclip.copy(latex_result)
        #     print("📋 Copied to clipboard!")
        # except ImportError:
        #     print("💡 Install pyperclip to auto-copy: pip install pyperclip")
    else:
        print("❌ Failed to generate LaTeX")

if __name__ == "__main__":
    main()