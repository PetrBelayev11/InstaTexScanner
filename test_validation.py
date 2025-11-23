"""
Script for testing the trained model on the validation set.
"""
import sys
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm
import numpy as np
from collections import Counter
try:
    import psutil
except ImportError:
    psutil = None
    print("⚠ psutil is not installed. Install it: pip install psutil")

# Add current directory to import path
sys.path.insert(0, str(Path(__file__).parent))

# Import modules
import importlib.util
spec = importlib.util.spec_from_file_location("model_im2markup", "code/models/model_im2markup.py")
model_im2markup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model_im2markup)
Im2Latex = model_im2markup.Im2Latex

spec = importlib.util.spec_from_file_location("preprocess", "code/models/preprocess.py")
preprocess = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preprocess)
preprocess_for_model = preprocess.preprocess_for_model

spec = importlib.util.spec_from_file_location("vocab", "code/vocab.py")
vocab_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vocab_module)
build_vocab = vocab_module.build_vocab


class Im2LatexDataset(Dataset):
    """Dataset for testing the im2latex model."""
    
    def __init__(self, data_list, vocab, max_len=256, cache_images=False, preload_images=False):
        self.data = data_list
        self.vocab = vocab
        self.max_len = max_len
        self.sos_token = vocab.get('<SOS>', 0)
        self.eos_token = vocab.get('<EOS>', 1)
        self.pad_token = vocab.get('<PAD>', 2)
        self.unk_token = vocab.get('<UNK>', 3)
        self.cache_images = cache_images
        self.preload_images = preload_images
        self._image_cache = {} if cache_images else None
        self._preloaded_images = None
        
        # Preload all images into memory
        if preload_images:
            print(f"\nPreloading {len(data_list)} images into memory...")
            self._preloaded_images = {}
            pbar = tqdm(data_list, desc="Loading images", ncols=100, file=sys.stdout)
            for idx, item in enumerate(pbar):
                img_path = item['img_path']
                try:
                    img = preprocess_for_model(img_path)
                    img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0)
                    self._preloaded_images[img_path] = img_tensor
                except Exception as e:
                    if idx < 10:
                        print(f"\nError loading {img_path}: {e}")
                    self._preloaded_images[img_path] = torch.zeros(1, 64, 256, dtype=torch.float32)
            print(f"✓ Loaded {len(self._preloaded_images)} images into memory ({len(self._preloaded_images) * 65 / 1024:.2f} MB)")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        img_path = item['img_path']
        latex = item['latex']
        
        # Preprocess image (with preloading or caching)
        if self._preloaded_images is not None:
            # Image already in memory - very fast!
            img_tensor = self._preloaded_images.get(img_path, torch.zeros(1, 64, 256, dtype=torch.float32))
        elif self.cache_images and img_path in self._image_cache:
            img_tensor = self._image_cache[img_path]
        else:
            # Load from disk (slow)
            try:
                img = preprocess_for_model(img_path)
                img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0)  # 1 x H x W
                if self.cache_images:
                    self._image_cache[img_path] = img_tensor
            except Exception as e:
                if idx < 10:  # Show only first 10 errors
                    print(f"Error loading image {img_path}: {e}")
                img_tensor = torch.zeros(1, 64, 256, dtype=torch.float32)
        
        # Convert LaTeX to token sequence
        tokens = [self.sos_token]
        tokens.extend(self.vocab.get(char, self.unk_token) for char in latex)
        tokens.append(self.eos_token)
        
        # Pad or truncate to max_len
        if len(tokens) > self.max_len:
            tokens = tokens[:self.max_len]
            tokens[-1] = self.eos_token
        else:
            tokens.extend([self.pad_token] * (self.max_len - len(tokens)))
        
        tokens_tensor = torch.tensor(tokens, dtype=torch.long)
        
        # Create target sequence (shifted by one for teacher forcing)
        target = tokens_tensor[1:].clone()
        input_seq = tokens_tensor[:-1]
        
        return img_tensor, input_seq, target, latex


def collate_fn(batch):
    """Collate function for DataLoader."""
    imgs, input_seqs, targets, latex_texts = zip(*batch)
    imgs = torch.stack(imgs)
    input_seqs = torch.stack(input_seqs)
    targets = torch.stack(targets)
    return imgs, input_seqs, targets, latex_texts


def tokens_to_text(tokens, idx_to_char, eos_token=1, pad_token=2):
    """Converts a sequence of tokens into text."""
    text = []
    for token in tokens:
        if token == eos_token:
            break
        if token == pad_token:
            continue
        if token in idx_to_char:
            text.append(idx_to_char[token])
    return ''.join(text)


def calculate_accuracy(pred_tokens, target_tokens, pad_token=2, eos_token=1):
    """Calculates token-level accuracy."""
    # Remove padding and EOS for comparison
    pred_clean = [t for t in pred_tokens if t != pad_token and t != eos_token]
    target_clean = [t for t in target_tokens if t != pad_token and t != eos_token]
    
    if len(target_clean) == 0:
        return 1.0 if len(pred_clean) == 0 else 0.0
    
    # Calculate accuracy by tokens
    min_len = min(len(pred_clean), len(target_clean))
    if min_len == 0:
        return 0.0
    
    correct = sum(1 for i in range(min_len) if pred_clean[i] == target_clean[i])
    return correct / len(target_clean)


def greedy_decode(model, img_tensor, vocab, max_len=256, sos_token=0, eos_token=1):
    """Greedy decoding for sequence generation.

    Accelerated version: one encoder pass per image and
    step-by-step decoder with attention, without full model forward
    at each step.
    """
    model.eval()
    device = next(model.parameters()).device
    img_tensor = img_tensor.to(device)

    decoded = [sos_token]

    with torch.no_grad():
        # Encode image once
        enc_feat = model.encoder(img_tensor)  # 1 x seq_len x C

        hidden = None

        for _ in range(max_len):
            last_token = torch.tensor([[decoded[-1]]], dtype=torch.long, device=device)  # 1 x 1

            # Embedding of the last token
            emb = model.decoder.embedding(last_token)  # 1 x 1 x H

            # Attention over image features
            query = model.decoder.query_proj(emb)              # 1 x 1 x enc_dim
            attn_output, _ = model.decoder.attention(
                query, enc_feat, enc_feat
            )                                                  # 1 x 1 x enc_dim

            # Input to GRU: embedding + context
            gru_input = torch.cat([emb, attn_output], dim=-1)  # 1 x 1 x (H + enc_dim)

            out, hidden = model.decoder.gru(gru_input, hidden)  # 1 x 1 x H
            logits = model.decoder.fc(out)                      # 1 x 1 x vocab

            next_token = logits.argmax(dim=-1).item()
            decoded.append(next_token)

            if next_token == eos_token:
                break

    # Remove SOS if it is first
    if decoded and decoded[0] == sos_token:
        decoded = decoded[1:]

    return decoded


def validate_model(model, dataloader, criterion, vocab, device, num_examples=5):
    """Validation of the model with metrics and examples output."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    total_accuracy = 0.0
    total_samples = 0
    
    # Create reverse vocabulary for decoding
    idx_to_char = {idx: char for char, idx in vocab.items()}
    eos_token = vocab.get('<EOS>', 1)
    pad_token = vocab.get('<PAD>', 2)
    
    examples = []
    example_count = 0
    
    # Use non_blocking to speed up transfer to GPU
    non_blocking = torch.cuda.is_available()
    
    with torch.no_grad():
        for batch_idx, (imgs, input_seqs, targets, latex_texts) in enumerate(tqdm(dataloader, desc="Validation", ncols=100)):
            # Asynchronous transfer to GPU for acceleration
            imgs = imgs.to(device, non_blocking=non_blocking)
            input_seqs = input_seqs.to(device, non_blocking=non_blocking)
            targets = targets.to(device, non_blocking=non_blocking)
            
            # Calculate loss and accuracy (teacher forcing) - FAST!
            logits = model(imgs, input_seqs)  # B x T x vocab
            logits_flat = logits.reshape(-1, logits.size(-1))
            targets_flat = targets.reshape(-1)
            
            mask = (targets_flat != pad_token)
            if mask.sum() > 0:
                loss = criterion(logits_flat[mask], targets_flat[mask])
                total_loss += loss.item()
                num_batches += 1
            
            # Calculate accuracy based on logits (fast, batch processing)
            pred_tokens_batch = logits.argmax(dim=-1).cpu().numpy()  # B x T
            targets_batch = targets.cpu().numpy()  # B x T
            
            batch_size = imgs.size(0)
            for i in range(batch_size):
                pred_seq = pred_tokens_batch[i]
                target_seq = targets_batch[i]
                
                # Calculate accuracy
                accuracy = calculate_accuracy(pred_seq, target_seq, pad_token, eos_token)
                total_accuracy += accuracy
                total_samples += 1
                
                # Greedy decode only for examples (slow, but needed only for output)
                if example_count < num_examples:
                    img = imgs[i:i+1]
                    pred_tokens = greedy_decode(model, img, vocab, max_len=256, 
                                               sos_token=vocab.get('<SOS>', 0), 
                                               eos_token=eos_token)
                    pred_text = tokens_to_text(pred_tokens, idx_to_char, eos_token, pad_token)
                    target_text = latex_texts[i]
                    examples.append({
                        'target': target_text,
                        'predicted': pred_text,
                        'accuracy': accuracy
                    })
                    example_count += 1
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    avg_accuracy = total_accuracy / total_samples if total_samples > 0 else 0.0
    
    return avg_loss, avg_accuracy, examples


def main():
    # Parameters
    json_path = Path("datasets/im2latex_prepared.json")
    model_path = Path("models/im2latex_best.pth")  # Use best model
    vocab_path = Path("models/vocab.json")
    batch_size = 32
    max_len = 256
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_examples = 10  # Number of examples to display
    
    print("=" * 80)
    print("TESTING MODEL ON VALIDATION SET")
    print("=" * 80)
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print()
    
    # Check for file existence
    if not json_path.exists():
        print(f"❌ Error: File {json_path} not found!")
        return
    
    if not model_path.exists():
        print(f"❌ Error: Model {model_path} not found!")
        print("Available models:")
        model_dir = Path("models")
        for pth_file in model_dir.glob("im2latex*.pth"):
            print(f"  - {pth_file}")
        return
    
    # Load data
    print(f"Loading data from {json_path}...")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    val_data = data.get('val', [])
    if not val_data:
        print("❌ Error: Validation set not found in JSON file!")
        print(f"Available keys: {list(data.keys())}")
        return
    
    print(f"✓ Loaded {len(val_data)} examples from validation set")
    print()
    
    # Load vocabulary
    print("Loading vocabulary...")
    if vocab_path.exists():
        vocab = build_vocab(vocab_file=str(vocab_path))
    else:
        vocab = build_vocab(json_path=str(json_path))
    vocab_size = len(vocab)
    print(f"✓ Vocabulary size: {vocab_size}")
    print()
    
    # Dataset and loader creation
    # Data loading strategies:
    # 1. preload_images=True - preload ALL images into RAM (fast, but requires a lot of RAM)
    # 2. cache_images=True - cache as loaded (medium speed)
    # 3. Both False - load from disk every time (slow, but saves RAM)
    
    # Check available RAM for preloading
    if psutil is not None:
        available_ram_gb = psutil.virtual_memory().available / 1024**3
    else:
        available_ram_gb = 16  # Assume 16GB if psutil is not available
    
    # Approximately 64*256*4 bytes per image = 65KB per image
    estimated_ram_gb = (len(val_data) * 65 * 1024) / 1024**3
    
    print(f"\nAvailable RAM: {available_ram_gb:.2f} GB")
    print(f"Estimated RAM for validation preload: {estimated_ram_gb:.2f} GB")
    
    # Automatic strategy selection
    if available_ram_gb > estimated_ram_gb * 1.5:  # Need 1.5x margin
        preload_images = True
        cache_images = False
        print(f"✓ Using preload of all images into RAM (fast!)")
    elif available_ram_gb > 8:
        preload_images = False
        cache_images = True
        print(f"✓ Using image caching")
    else:
        preload_images = False
        cache_images = False
        print(f"⚠ Loading from disk (requires more RAM for acceleration)")
    print()
    
    print("Creating dataset...")
    val_dataset = Im2LatexDataset(val_data, vocab, max_len=max_len, 
                                 cache_images=cache_images, preload_images=preload_images)
    
    # DataLoader settings with optimizations from training code
    use_pin_memory = torch.cuda.is_available()
    
    # If preloading is on, num_workers is not needed (data already in memory)
    num_workers = 0 if preload_images else 0  # For Windows and preloading
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=use_pin_memory,  # Accelerate transfer to GPU
        persistent_workers=num_workers > 0,  # Only if using workers
        prefetch_factor=2 if num_workers > 0 else None  # Only with workers
    )
    print(f"✓ Dataset created, batches: {len(val_loader)}")
    print()
    
    # Load model
    print(f"Loading model from {model_path}...")
    model = Im2Latex(vocab_size=vocab_size).to(device)
    
    try:
        checkpoint = torch.load(model_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✓ Model loaded (epoch {checkpoint.get('epoch', '?')})")
        else:
            model.load_state_dict(checkpoint)
            print("✓ Model loaded")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    print(f"✓ Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()
    
    # Loss function
    criterion = nn.CrossEntropyLoss(ignore_index=2)  # Ignore PAD token
    
    # Validation
    print("Starting validation...")
    print("-" * 80)
    val_loss, val_accuracy, examples = validate_model(
        model, val_loader, criterion, vocab, device, num_examples=num_examples
    )
    
    # Output results
    print()
    print("=" * 80)
    print("VALIDATION RESULTS")
    print("=" * 80)
    print(f"Average Loss: {val_loss:.4f}")
    print(f"Average Accuracy (token-level): {val_accuracy * 100:.2f}%")
    print()
    
    # Prediction examples
    print("=" * 80)
    print(f"PREDICTION EXAMPLES (first {len(examples)} from validation set)")
    print("=" * 80)
    for i, ex in enumerate(examples, 1):
        print(f"\nExample {i} (accuracy: {ex['accuracy']*100:.1f}%):")
        print(f"  Target LaTeX:  {ex['target'][:150]}{'...' if len(ex['target']) > 150 else ''}")
        print(f"  Predicted:     {ex['predicted'][:150]}{'...' if len(ex['predicted']) > 150 else ''}")
        if ex['target'] == ex['predicted']:
            print("  ✓ Full match!")
        else:
            # Show differences
            target_chars = list(ex['target'])
            pred_chars = list(ex['predicted'])
            min_len = min(len(target_chars), len(pred_chars))
            differences = sum(1 for j in range(min_len) if target_chars[j] != pred_chars[j])
            differences += abs(len(target_chars) - len(pred_chars))
            print(f"  Differences: {differences} chars")
    
    print()
    print("=" * 80)
    print("Validation completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()

