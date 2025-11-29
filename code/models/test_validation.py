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

# Add current directory to import path
sys.path.insert(0, str(Path(__file__).parent))

# Import modules
import importlib.util
spec = importlib.util.spec_from_file_location("model_im2markup", "model_im2markup.py")
model_im2markup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model_im2markup)
Im2Latex = model_im2markup.Im2Latex

spec = importlib.util.spec_from_file_location("preprocess", "preprocess.py")
preprocess = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preprocess)
preprocess_for_model = preprocess.preprocess_for_model

spec = importlib.util.spec_from_file_location("vocab", "vocab.py")
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
        
        if self._preloaded_images is not None:
            img_tensor = self._preloaded_images.get(img_path, torch.zeros(1, 64, 256, dtype=torch.float32))
        elif self.cache_images and img_path in self._image_cache:
            img_tensor = self._image_cache[img_path]
        else:
            try:
                img = preprocess_for_model(img_path)
                img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0)
                if self.cache_images:
                    self._image_cache[img_path] = img_tensor
            except Exception as e:
                if idx < 10:
                    print(f"Error loading image {img_path}: {e}")
                img_tensor = torch.zeros(1, 64, 256, dtype=torch.float32)
        
        tokens = [self.sos_token]
        tokens.extend(self.vocab.get(char, self.unk_token) for char in latex)
        tokens.append(self.eos_token)
        
        if len(tokens) > self.max_len:
            tokens = tokens[:self.max_len]
            tokens[-1] = self.eos_token
        else:
            tokens.extend([self.pad_token] * (self.max_len - len(tokens)))
        
        tokens_tensor = torch.tensor(tokens, dtype=torch.long)
        target = tokens_tensor[1:].clone()
        input_seq = tokens_tensor[:-1]
        
        return img_tensor, input_seq, target, latex


def collate_fn(batch):
    imgs, input_seqs, targets, latex_texts = zip(*batch)
    imgs = torch.stack(imgs)
    input_seqs = torch.stack(input_seqs)
    targets = torch.stack(targets)
    return imgs, input_seqs, targets, latex_texts


def tokens_to_text(tokens, idx_to_char, eos_token=1, pad_token=2):
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
    pred_clean = [t for t in pred_tokens if t != pad_token and t != eos_token]
    target_clean = [t for t in target_tokens if t != pad_token and t != eos_token]
    
    if len(target_clean) == 0:
        return 1.0 if len(pred_clean) == 0 else 0.0
    
    min_len = min(len(pred_clean), len(target_clean))
    if min_len == 0:
        return 0.0
    
    correct = sum(1 for i in range(min_len) if pred_clean[i] == target_clean[i])
    return correct / len(target_clean)


def working_beam_search(model, img_tensor, vocab, beam_size=5, max_len=256):
    """Proper beam search that works with your encoder-decoder architecture"""
    model.eval()
    device = next(model.parameters()).device
    img_tensor = img_tensor.to(device)
    
    sos_token = vocab.get('<SOS>', 0)
    eos_token = vocab.get('<EOS>', 1)
    
    with torch.no_grad():
        encoder_output = model.encoder(img_tensor)
        
        beams = [([sos_token], 0.0)]
        completed = []
        
        for step in range(max_len):
            new_beams = []
            
            for seq, score in beams:
                if seq[-1] == eos_token:
                    completed.append((seq, score))
                    continue
                
                decoder_input = torch.tensor([seq], device=device)
                logits = model.decoder(encoder_output, decoder_input)
                next_token_logits = logits[0, -1, :]
                next_token_probs = torch.softmax(next_token_logits, dim=-1)
                
                top_probs, top_indices = torch.topk(next_token_probs, beam_size)
                
                for i in range(beam_size):
                    token = top_indices[i].item()
                    new_seq = seq + [token]
                    new_score = score + torch.log(top_probs[i]).item()
                    new_beams.append((new_seq, new_score))
            
            if not new_beams:
                break
                
            beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_size]
            
            still_active = []
            for seq, score in beams:
                if seq[-1] == eos_token:
                    completed.append((seq, score))
                else:
                    still_active.append((seq, score))
            beams = still_active
            
            if not beams:
                break
        
        all_candidates = completed + beams
        if not all_candidates:
            return [sos_token, eos_token]
        
        best_seq = max(all_candidates, key=lambda x: x[1] / len(x[0]))[0]
        return best_seq[1:]


def diverse_beam_search(model, img_tensor, vocab, beam_size=3, num_groups=2, max_len=256):
    """Diverse beam search to avoid repetitive outputs"""
    model.eval()
    device = next(model.parameters()).device
    img_tensor = img_tensor.to(device)
    
    sos_token = vocab.get('<SOS>', 0)
    eos_token = vocab.get('<EOS>', 1)
    
    with torch.no_grad():
        encoder_output = model.encoder(img_tensor)
        
        beam_groups = [[] for _ in range(num_groups)]
        beam_groups[0] = [([sos_token], 0.0)]
        
        completed = []
        
        for step in range(max_len):
            next_groups = [[] for _ in range(num_groups)]
            
            for group_idx in range(num_groups):
                for seq, score in beam_groups[group_idx]:
                    if seq[-1] == eos_token:
                        completed.append((seq, score, group_idx))
                        continue
                    
                    decoder_input = torch.tensor([seq], device=device)
                    logits = model.decoder(encoder_output, decoder_input)
                    next_token_logits = logits[0, -1, :]
                    next_token_probs = torch.softmax(next_token_logits, dim=-1)
                    
                    top_probs, top_indices = torch.topk(next_token_probs, beam_size)
                    
                    for i in range(beam_size):
                        token = top_indices[i].item()
                        new_seq = seq + [token]
                        new_score = score + torch.log(top_probs[i]).item()
                        next_groups[group_idx].append((new_seq, new_score))
            
            for group_idx in range(num_groups):
                if next_groups[group_idx]:
                    beam_groups[group_idx] = sorted(next_groups[group_idx], 
                                                   key=lambda x: x[1], reverse=True)[:beam_size]
                else:
                    beam_groups[group_idx] = []
            
            if all(not group for group in beam_groups):
                break
        
        all_candidates = completed + [
            (seq, score, group_idx) 
            for group_idx, group in enumerate(beam_groups) 
            for seq, score in group
        ]
        
        if not all_candidates:
            return [sos_token, eos_token]
        
        best_seq = max(all_candidates, key=lambda x: x[1] / max(1, len(x[0])))[0]
        return best_seq[1:]


def validate_with_beam_search(model, dataloader, criterion, vocab, device, num_examples=5):
    """Validation using proper beam search"""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    total_accuracy = 0.0
    total_samples = 0
    
    idx_to_char = {idx: char for char, idx in vocab.items()}
    eos_token = vocab.get('<EOS>', 1)
    pad_token = vocab.get('<PAD>', 2)
    
    examples = []
    
    with torch.no_grad():
        for batch_idx, (imgs, input_seqs, targets, latex_texts) in enumerate(tqdm(dataloader, desc="Validation")):
            imgs = imgs.to(device)
            input_seqs = input_seqs.to(device)
            targets = targets.to(device)
            
            logits = model(imgs, input_seqs)
            logits_flat = logits.reshape(-1, logits.size(-1))
            targets_flat = targets.reshape(-1)
            
            mask = (targets_flat != pad_token)
            if mask.sum() > 0:
                loss = criterion(logits_flat[mask], targets_flat[mask])
                total_loss += loss.item()
                num_batches += 1
            
            pred_tokens_batch = logits.argmax(dim=-1).cpu().numpy()
            targets_batch = targets.cpu().numpy()
            
            batch_size = imgs.size(0)
            for i in range(batch_size):
                pred_seq = pred_tokens_batch[i]
                target_seq = targets_batch[i]
                
                accuracy = calculate_accuracy(pred_seq, target_seq, pad_token, eos_token)
                total_accuracy += accuracy
                total_samples += 1
                
                if len(examples) < num_examples:
                    img = imgs[i:i+1]
                    
                    beam_result = working_beam_search(model, img, vocab, beam_size=5, max_len=256)
                    beam_text = tokens_to_text(beam_result, idx_to_char, eos_token, pad_token)
                    
                    diverse_result = diverse_beam_search(model, img, vocab, beam_size=3, num_groups=2, max_len=256)
                    diverse_text = tokens_to_text(diverse_result, idx_to_char, eos_token, pad_token)
                    
                    pred_text = tokens_to_text(pred_seq, idx_to_char, eos_token, pad_token)
                    target_text = latex_texts[i]
                    
                    examples.append({
                        'target': target_text,
                        'teacher_forcing': pred_text,
                        'beam_search': beam_text,
                        'diverse_beam': diverse_text,
                        'accuracy': accuracy
                    })
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    avg_accuracy = total_accuracy / total_samples if total_samples > 0 else 0.0
    
    return avg_loss, avg_accuracy, examples

def debug_beam_search_step(model, img_tensor, vocab, max_steps=10):
    """Debug what happens in the first few steps of beam search"""
    model.eval()
    device = next(model.parameters()).device
    img_tensor = img_tensor.to(device)
    
    sos_token = vocab.get('<SOS>', 0)
    idx_to_char = {idx: char for char, idx in vocab.items()}
    
    with torch.no_grad():
        encoder_output = model.encoder(img_tensor)
        print(f"Encoder output - mean: {encoder_output.mean().item():.3f}, std: {encoder_output.std().item():.3f}")
        print(f"Encoder output range: [{encoder_output.min().item():.3f}, {encoder_output.max().item():.3f}]")
        
        # Start with SOS
        current_seq = torch.tensor([[sos_token]], device=device)
        
        for step in range(max_steps):
            logits = model.decoder(encoder_output, current_seq)
            next_token_logits = logits[0, -1, :]
            next_token_probs = torch.softmax(next_token_logits, dim=-1)
            
            # Check entropy - if it's low, model is very confident (overfitting)
            entropy = -torch.sum(next_token_probs * torch.log(next_token_probs + 1e-9))
            print(f"\nStep {step}:")
            print(f"  Current sequence: {[idx_to_char.get(t, '?') for t in current_seq[0].tolist()]}")
            print(f"  Entropy: {entropy.item():.3f} (low = overconfident)")
            
            # Show top predictions
            top_probs, top_indices = torch.topk(next_token_probs, 5)
            for i, (idx, prob) in enumerate(zip(top_indices, top_probs)):
                char = idx_to_char.get(idx.item(), '?')
                print(f"    {i+1}. '{char}' (prob: {prob.item():.3f})")
            
            next_token = next_token_logits.argmax().item()
            current_seq = torch.cat([current_seq, torch.tensor([[next_token]], device=device)], dim=1)
            
            if next_token == vocab.get('<EOS>', 1):
                print("  EOS generated")
                break

def check_model_training_state(model, dataloader, device):
    """Verify the model learned anything at all"""
    model.eval()
    with torch.no_grad():
        # Test on a few samples
        for imgs, input_seqs, targets, _ in dataloader:
            imgs = imgs.to(device)
            input_seqs = input_seqs.to(device)
            
            logits = model(imgs, input_seqs)
            preds = logits.argmax(dim=-1)
            
            # Check if predictions are random or consistent
            unique_preds = torch.unique(preds)
            print(f"Unique predictions in batch: {len(unique_preds)}/{logits.size(-1)}")
            
            # Check if model outputs same thing for different inputs
            if imgs.size(0) > 1:
                first_pred = preds[0]
                same_count = sum(torch.all(preds[i] == first_pred) for i in range(1, imgs.size(0)))
                print(f"Identical predictions across batch: {same_count}/{imgs.size(0)-1}")
            break

def temperature_sampling_decode(model, img_tensor, vocab, temperature=1.0, max_len=256):
    """Temperature sampling - more diverse than greedy"""
    model.eval()
    device = next(model.parameters()).device
    img_tensor = img_tensor.to(device)
    
    sos_token = vocab.get('<SOS>', 0)
    eos_token = vocab.get('<EOS>', 1)
    
    with torch.no_grad():
        encoder_output = model.encoder(img_tensor)
        current_seq = torch.tensor([[sos_token]], device=device)
        decoded = []
        
        for _ in range(max_len):
            logits = model.decoder(encoder_output, current_seq)
            next_token_logits = logits[0, -1, :] / temperature
            next_token_probs = torch.softmax(next_token_logits, dim=-1)
            
            # Sample from distribution
            next_token = torch.multinomial(next_token_probs, 1).item()
            decoded.append(next_token)
            
            if next_token == eos_token:
                break
                
            current_seq = torch.cat([current_seq, torch.tensor([[next_token]], device=device)], dim=1)
        
        return decoded

def test_image_attention(model, vocab, device):
    """Test if model actually uses image information"""
    print("=" * 80)
    print("TESTING IF MODEL LOOKS AT IMAGES")
    print("=" * 80)
    
    model.eval()
    idx_to_char = {idx: char for char, idx in vocab.items()}
    
    # Test 1: Same image multiple times
    print("\n1. SAME IMAGE TEST:")
    test_img = torch.randn(1, 1, 64, 256).to(device)  # Random noise
    for i in range(3):
        result = working_beam_search(model, test_img, vocab)
        text = tokens_to_text(result, idx_to_char)
        print(f"  Run {i+1}: {text[:50]}...")
    
    # Test 2: Different images
    print("\n2. DIFFERENT IMAGES TEST:")
    for i in range(3):
        different_img = torch.randn(1, 1, 64, 256).to(device)  # Different random noise
        result = working_beam_search(model, different_img, vocab)
        text = tokens_to_text(result, idx_to_char)
        print(f"  Image {i+1}: {text[:50]}...")
    
    # Test 3: All-zero image vs all-one image
    print("\n3. EXTREME IMAGES TEST:")
    zero_img = torch.zeros(1, 1, 64, 256).to(device)
    ones_img = torch.ones(1, 1, 64, 256).to(device)
    
    zero_result = working_beam_search(model, zero_img, vocab)
    ones_result = working_beam_search(model, ones_img, vocab)
    
    print(f"  All zeros: {tokens_to_text(zero_result, idx_to_char)[:50]}...")
    print(f"  All ones:  {tokens_to_text(ones_result, idx_to_char)[:50]}...")
    
    # Test 4: Real images from dataset
    print("\n4. REAL DATASET IMAGES TEST:")
    dataset = Im2LatexDataset(val_data[:5], vocab, max_len=256, preload_images=False)
    for i in range(3):
        img, _, _, latex = dataset[i]
        result = working_beam_search(model, img.unsqueeze(0).to(device), vocab)
        text = tokens_to_text(result, idx_to_char)
        print(f"  Real image {i+1}:")
        print(f"    Target: {latex[:50]}...")
        print(f"    Predicted: {text[:50]}...")

def debug_encoder_outputs(model, device):
    """See what the encoder actually outputs"""
    print("\n" + "=" * 80)
    print("ENCODER OUTPUT ANALYSIS")
    print("=" * 80)
    
    model.eval()
    
    # Test with different inputs
    test_cases = [
        ("Zeros", torch.zeros(1, 1, 64, 256).to(device)),
        ("Ones", torch.ones(1, 1, 64, 256).to(device)),
        ("Random", torch.randn(1, 1, 64, 256).to(device)),
    ]
    
    for name, img in test_cases:
        with torch.no_grad():
            enc_out = model.encoder(img)
            print(f"\n{name} input:")
            print(f"  Encoder output - mean: {enc_out.mean().item():.6f}")
            print(f"  Encoder output - std:  {enc_out.std().item():.6f}")
            print(f"  Encoder output - min:  {enc_out.min().item():.6f}")
            print(f"  Encoder output - max:  {enc_out.max().item():.6f}")
            
            # Check if outputs are identical across different inputs
            if name == "Zeros":
                zeros_enc = enc_out
            elif name == "Ones":
                ones_enc = enc_out
                diff = (zeros_enc - ones_enc).abs().mean().item()
                print(f"  Difference vs zeros: {diff:.6f}")

def test_encoder_gradients(model, device):
    """Check if encoder gradients are flowing during training"""
    print("\n" + "=" * 80)
    print("ENCODER GRADIENT FLOW TEST")
    print("=" * 80)
    
    model.train()
    criterion = nn.CrossEntropyLoss()
    
    # Create a simple training step
    img = torch.randn(2, 1, 64, 256).to(device)  # Batch of 2
    # Simple sequences: "x=y" and "a=b"
    input_seqs = torch.tensor([
        [vocab.get(c, 0) for c in "x=y"] + [2] * 253,  # "<SOS>x=y" padded
        [vocab.get(c, 0) for c in "a=b"] + [2] * 253,
    ]).to(device)
    targets = torch.tensor([
        [vocab.get(c, 0) for c in "x=y<EOS>"] + [2] * 252,  # "x=y<EOS>" padded  
        [vocab.get(c, 0) for c in "a=b<EOS>"] + [2] * 252,
    ]).to(device)
    
    # Forward pass
    logits = model(img, input_seqs)
    loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
    
    # Backward pass
    model.zero_grad()
    loss.backward()
    
    # Check encoder gradients
    enc_grad_norm = 0.0
    enc_param_count = 0
    for name, param in model.encoder.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            enc_grad_norm += grad_norm
            enc_param_count += 1
            print(f"  {name}: grad norm = {grad_norm:.8f}")
    
    # Check decoder gradients  
    dec_grad_norm = 0.0
    for param in model.decoder.parameters():
        if param.grad is not None:
            dec_grad_norm += param.grad.norm().item()
    
    print(f"\nEncoder total grad norm: {enc_grad_norm:.8f}")
    print(f"Decoder total grad norm: {dec_grad_norm:.8f}")
    print(f"Encoder/Decoder grad ratio: {enc_grad_norm/dec_grad_norm:.8f}")
    
    if enc_grad_norm < 1e-10:
        print("🚨 CRITICAL: ENCODER GRADIENTS ARE ZERO!")
        print("The encoder is not learning - it's dead!")

def main():
    json_path = Path("../../datasets/im2latex_prepared.json")
    model_path = Path("../../models/im2latex_best.pth")
    vocab_path = Path("models/vocab.json")
    batch_size = 32
    max_len = 256
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_examples = 10
    
    print("=" * 80)
    print("TESTING MODEL ON VALIDATION SET")
    print("=" * 80)
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print()
    
    if not json_path.exists():
        print(f"❌ Error: File {json_path} not found!")
        return
    
    if not model_path.exists():
        print(f"❌ Error: Model {model_path} not found!")
        model_dir = Path("models")
        for pth_file in model_dir.glob("im2latex*.pth"):
            print(f"  - {pth_file}")
        return
    
    print(f"Loading data from {json_path}...")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    val_data = data.get('val', [])
    if not val_data:
        print("❌ Error: Validation set not found in JSON file!")
        return
    
    print(f"✓ Loaded {len(val_data)} examples from validation set")
    print()
    
    print("Loading vocabulary...")
    if vocab_path.exists():
        vocab = build_vocab(vocab_file=str(vocab_path))
    else:
        vocab = build_vocab(json_path=str(json_path))
    vocab_size = len(vocab)
    print(f"✓ Vocabulary size: {vocab_size}")
    print()
    
    if psutil is not None:
        available_ram_gb = psutil.virtual_memory().available / 1024**3
    else:
        available_ram_gb = 16
    
    estimated_ram_gb = (len(val_data) * 65 * 1024) / 1024**3
    
    print(f"\nAvailable RAM: {available_ram_gb:.2f} GB")
    print(f"Estimated RAM for validation preload: {estimated_ram_gb:.2f} GB")
    
    if available_ram_gb > estimated_ram_gb * 1.5:
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
        print(f"⚠ Loading from disk")
    print()
    
    print("Creating dataset...")
    val_dataset = Im2LatexDataset(val_data, vocab, max_len=max_len, 
                                 cache_images=cache_images, preload_images=preload_images)
    
    use_pin_memory = torch.cuda.is_available()
    num_workers = 0 if preload_images else 0
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None
    )
    print(f"✓ Dataset created, batches: {len(val_loader)}")
    print()
    
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
    
    criterion = nn.CrossEntropyLoss(ignore_index=2)
    
    print("Starting validation with beam search...")
    print("-" * 80)
    val_loss, val_accuracy, examples = validate_with_beam_search(
        model, val_loader, criterion, vocab, device, num_examples=num_examples
    )
    
    print()
    print("=" * 80)
    print("VALIDATION RESULTS")
    print("=" * 80)
    print(f"Average Loss: {val_loss:.4f}")
    print(f"Average Accuracy (token-level): {val_accuracy * 100:.2f}%")
    print()
    
    print("=" * 80)
    print(f"PREDICTION EXAMPLES (first {len(examples)} from validation set)")
    print("=" * 80)
    for i, ex in enumerate(examples, 1):
        print(f"\nExample {i} (teacher-forcing accuracy: {ex['accuracy']*100:.1f}%):")
        print(f"  Target LaTeX:  {ex['target'][:100]}{'...' if len(ex['target']) > 100 else ''}")
        print(f"  Teacher-forcing: {ex['teacher_forcing'][:100]}{'...' if len(ex['teacher_forcing']) > 100 else ''}")
        print(f"  Beam search:     {ex['beam_search'][:100]}{'...' if len(ex['beam_search']) > 100 else ''}")
        print(f"  Diverse beam:    {ex['diverse_beam'][:100]}{'...' if len(ex['diverse_beam']) > 100 else ''}")
        
        if ex['teacher_forcing'] == ex['target']:
            print("  ✓ Teacher-forcing: Full match!")
        else:
            target_chars = list(ex['target'])
            pred_chars = list(ex['teacher_forcing'])
            min_len = min(len(target_chars), len(pred_chars))
            differences = sum(1 for j in range(min_len) if target_chars[j] != pred_chars[j])
            differences += abs(len(target_chars) - len(pred_chars))
            print(f"  Teacher-forcing differences: {differences} chars")
        
        if ex['beam_search'] == ex['target']:
            print("  ✓ Beam search: Full match!")
        elif len(ex['beam_search']) > 10:
            print(f"  Beam search length: {len(ex['beam_search'])} chars")
    
    print()
    print("=" * 80)
    print("Validation completed!")
    print("=" * 80)
    
    print("\n" + "=" * 80)
    print("COMPREHENSIVE MODEL DIAGNOSTICS")
    print("=" * 80)

    test_image_attention(model, vocab, device)
    debug_encoder_outputs(model, device) 
    test_encoder_gradients(model, device)
    



if __name__ == "__main__":
    main()
    