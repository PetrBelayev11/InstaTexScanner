"""
Script for training the im2latex model on prepared data.
"""
import sys
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm
import numpy as np
from PIL import Image
import cv2
import multiprocessing
try:
    import psutil
except ImportError:
    psutil = None
    print("⚠ psutil is not installed. Install it: pip install psutil")

# Add current directory to import path
sys.path.insert(0, str(Path(__file__).parent))

# Import modules directly, avoiding conflict with stdlib 'code'
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
    """Dataset for training im2latex model."""
    
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
                    if img.shape != (128, 512):
                        print(f"⚠ Warning: Image {img_path} has shape {img.shape}, expected (128, 512)")
                    img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0)
                    self._preloaded_images[img_path] = img_tensor
                except Exception as e:
                    if idx < 10:
                        print(f"\nError loading {img_path}: {e}")
                    self._preloaded_images[img_path] = torch.zeros(1, 128, 512, dtype=torch.float32)
            print(f"✓ Loaded {len(self._preloaded_images)} images into memory")
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        img_path = item['img_path']
        latex = item['latex']
        
        # Preprocess image
        if self._preloaded_images is not None:
            img_tensor = self._preloaded_images.get(img_path, torch.zeros(1, 128, 512, dtype=torch.float32))
        elif self.cache_images and img_path in self._image_cache:
            img_tensor = self._image_cache[img_path]
        else:
            try:
                img = preprocess_for_model(img_path)
                # VERIFY SIZE
                if img.shape != (128, 512):
                    print(f"⚠ Size mismatch: {img_path} has shape {img.shape}")
                img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0)
                if self.cache_images:
                    self._image_cache[img_path] = img_tensor
            except Exception as e:
                if idx < 10:
                    print(f"Error loading image {img_path}: {e}")
                img_tensor = torch.zeros(1, 128, 512, dtype=torch.float32)
        
        # Convert LaTeX to token sequence (optimized)
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
        
        return img_tensor, input_seq, target


def collate_fn(batch):
    """Collate function for DataLoader."""
    imgs, input_seqs, targets = zip(*batch)
    imgs = torch.stack(imgs)
    input_seqs = torch.stack(input_seqs)
    targets = torch.stack(targets)
    return imgs, input_seqs, targets


def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """One training epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    # Use non_blocking to speed up transfer to GPU
    non_blocking = torch.cuda.is_available()
    
    # Cleaner progress bar with minimal info
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}", ncols=80, file=sys.stdout)
    
    for imgs, input_seqs, targets in pbar:
        # Asynchronous transfer to GPU for acceleration
        imgs = imgs.to(device, non_blocking=non_blocking)
        input_seqs = input_seqs.to(device, non_blocking=non_blocking)
        targets = targets.to(device, non_blocking=non_blocking)
        
        # Forward pass
        optimizer.zero_grad()
        logits = model(imgs, input_seqs)  # B x T x vocab
        
        # Reshape for loss calculation
        logits = logits.reshape(-1, logits.size(-1))
        targets = targets.reshape(-1)
        
        # Calculate loss (ignore padding tokens)
        mask = (targets != 2)  # PAD token = 2
        if mask.sum() > 0:
            loss = criterion(logits[mask], targets[mask])
        else:
            continue
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        # Update progress bar with cleaner formatting
        avg_loss = total_loss / num_batches
        postfix = {'loss': f'{loss.item():.3f}', 'avg': f'{avg_loss:.3f}'}
        
        # Add GPU info less frequently to reduce clutter
        if torch.cuda.is_available() and num_batches % 20 == 0:
            gpu_memory = torch.cuda.memory_allocated(0) / 1024**3
            postfix['GPU'] = f'{gpu_memory:.1f}GB'
        
        pbar.set_postfix(postfix)
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss


def validate(model, dataloader, criterion, device):
    """Model validation."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for imgs, input_seqs, targets in tqdm(dataloader, desc="Validation", ncols=100, leave=False):
            imgs = imgs.to(device)
            input_seqs = input_seqs.to(device)
            targets = targets.to(device)
            
            logits = model(imgs, input_seqs)
            logits = logits.reshape(-1, logits.size(-1))
            targets = targets.reshape(-1)
            
            mask = (targets != 2)  # PAD token
            if mask.sum() > 0:
                loss = criterion(logits[mask], targets[mask])
                total_loss += loss.item()
                num_batches += 1
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss


def main():
    # Training parameters
    json_path = Path("../../datasets/im2latex_prepared.json")
    batch_size = 16  # Optimal for GTX 1080 (8GB)
    # Increase number of epochs for model fine-tuning
    num_epochs = 30
    learning_rate = 1e-4
    max_len = 256
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    print(f"Loading data from {json_path}...")
    
    # Load data
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    train_data = data.get('train', [])
    val_data = data.get('val', [])
    
    print(f"Train samples: {len(train_data)}")
    print(f"Val samples: {len(val_data)}")
    
    # Build vocabulary
    print("Building vocabulary...")
    vocab = build_vocab(json_path=str(json_path))
    vocab_size = len(vocab)
    print(f"Vocabulary size: {vocab_size}")
    
    # Create datasets
    # Data loading strategies:
    # 1. preload_images=True - preload ALL images into RAM (fast, but requires a lot of RAM)
    # 2. cache_images=True - cache as loaded (medium speed)
    # 3. Both False - load from disk every time (slow, but saves RAM)
    
    # Check available RAM for preloading
    if psutil is not None:
        available_ram_gb = psutil.virtual_memory().available / 1024**3
    else:
        available_ram_gb = 16  # Assume 16GB if psutil is not available
    total_samples = len(train_data) + (len(val_data) if val_data else 0)
    # Approximately 64*256*4 bytes per image = 65KB per image
    estimated_ram_gb = (total_samples * 65 * 1024 * 4) / 1024**3  # 4x larger images
    print(f"\nAvailable RAM: {available_ram_gb:.2f} GB")
    print(f"Estimated RAM for preload: {estimated_ram_gb:.2f} GB (4x larger due to 128x512 images)")
    
    # Automatic strategy selection
    if available_ram_gb > estimated_ram_gb * 1.5:  # Need 1.5x margin
        preload_images = True
        cache_images = False
        print(f"✓ Using preload of all images into RAM (fast!)")
    elif available_ram_gb > 16:
        preload_images = False
        cache_images = True
        print(f"✓ Using image caching")
    else:
        preload_images = False
        cache_images = False
        print(f"⚠ Loading from disk (requires more RAM for acceleration)")
    
    train_dataset = Im2LatexDataset(train_data, vocab, max_len=max_len, 
                                     cache_images=cache_images, preload_images=preload_images)
    val_dataset = Im2LatexDataset(val_data, vocab, max_len=max_len, 
                                  cache_images=cache_images, preload_images=preload_images) if val_data else None
    
    # DataLoader settings
    # Try to use multiprocessing on Windows for parallel loading
    use_pin_memory = torch.cuda.is_available()
    
    # On Windows num_workers > 0 can be used, but needs proper configuration
    # Check if multiprocessing works
    try:
        # Test multiprocessing on Windows
        if sys.platform == 'win32':
            # On Windows if __name__ == '__main__' is needed for multiprocessing
            # Try using 2-4 workers
            num_workers = min(4, multiprocessing.cpu_count())
            print(f"\nAttempting to use {num_workers} workers for parallel loading...")
        else:
            num_workers = min(4, multiprocessing.cpu_count())
    except:
        num_workers = 0
        print("\n⚠ Multiprocessing unavailable, using num_workers=0")
    
    # If preloading is enabled, num_workers is not needed (data already in memory)
    if preload_images:
        num_workers = 0
        print("  (num_workers=0, as data is preloaded in memory)")
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=use_pin_memory,  # Accelerate transfer to GPU
        persistent_workers=num_workers > 0,  # Only if using workers
        prefetch_factor=2 if num_workers > 0 else None  # Only with workers
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None
    ) if val_dataset else None
    
    # Create model
    print("Creating model...")
    model = Im2Latex(vocab_size=vocab_size).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Check GPU usage
    if torch.cuda.is_available():
        # Test run to check GPU
        print("\nChecking GPU usage...")
        model.eval()
        with torch.no_grad():
            test_img = torch.randn(1, 1, 64, 256).to(device)
            test_seq = torch.randint(0, vocab_size, (1, 255)).to(device)
            _ = model(test_img, test_seq)
        print(f"✓ GPU works! Memory used: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
        torch.cuda.reset_peak_memory_stats()
        model.train()
    
    # Optimizer, loss function and learning rate scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=2)  # Ignore PAD token
    # Scheduler: reduces learning rate if validation loss stops improving.
    # In your PyTorch version the verbose argument may not be available, so we don't use it.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
    )
    
    # Directory for saving models
    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)

    # Training
    print("\nStarting training...")
    best_val_loss = float('inf')
    
    # Check existing checkpoints for recovery
    start_epoch = 1
    for epoch in range(num_epochs, 0, -1):
        checkpoint_path = model_dir / f"im2latex_epoch{epoch}.pth"
        if checkpoint_path.exists():
            print(f"\nFound checkpoint for epoch {epoch}. Loading...")
            try:
                checkpoint = torch.load(checkpoint_path, map_location=device)
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    # New format with full checkpoint
                    model.load_state_dict(checkpoint['model_state_dict'])
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    best_val_loss = checkpoint.get('best_val_loss', float('inf'))
                    print(f"Loaded checkpoint: epoch {checkpoint.get('epoch', epoch)}, best_val_loss={best_val_loss:.4f}")
                else:
                    # Old format (state_dict only)
                    model.load_state_dict(checkpoint)
                    print("Loaded old checkpoint format (without optimizer)")
                start_epoch = epoch + 1
                print(f"Resuming training from epoch {start_epoch}")
                break
            except Exception as e:
                print(f"Error loading checkpoint: {e}. Starting from scratch.")
                break
    
    for epoch in range(start_epoch, num_epochs + 1):
        # Training
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        print(f"Epoch {epoch}/{num_epochs} - Train Loss: {train_loss:.4f}")
        
        # Validation
        if val_loader:
            val_loss = validate(model, val_loader, criterion, device)
            print(f"Epoch {epoch}/{num_epochs} - Val Loss: {val_loss:.4f}")
            # Update scheduler based on validation loss
            scheduler.step(val_loss)

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_path = model_dir / f"im2latex_best.pth"
                torch.save(model.state_dict(), best_model_path)
                print(f"Saved best model to {best_model_path}")
        else:
            # If no validation, use training loss
            scheduler.step(train_loss)
        
        # Save checkpoint every epoch (with optimizer for full recovery)
        checkpoint_path = model_dir / f"im2latex_epoch{epoch}.pth"
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
            'vocab_size': vocab_size
        }
        torch.save(checkpoint, checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")
        
        # Save vocabulary
        vocab_path = model_dir / "vocab.json"
        with open(vocab_path, 'w', encoding='utf-8') as f:
            json.dump(vocab, f, ensure_ascii=False, indent=2)
    
    print("\nTraining completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Models saved to {model_dir}")


if __name__ == "__main__":
    main()

