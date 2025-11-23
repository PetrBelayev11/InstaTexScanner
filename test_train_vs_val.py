"""
Script for evaluating the trained model on training and validation sets
using optimized run code (same as in test_validation.py).
"""
import sys
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm

try:
    import psutil
except ImportError:
    psutil = None
    print("⚠ psutil is not installed. Install it: pip install psutil")

# Add current directory to import path
sys.path.insert(0, str(Path(__file__).parent))

# Import modules via importlib, as in other scripts
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
    """Dataset for evaluating im2latex model (train/val)."""

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

        # Preload all images into memory (as in test_validation.py)
        if preload_images:
            print(f"\nPreloading {len(data_list)} images into memory...")
            from tqdm import tqdm as tqdm_local

            self._preloaded_images = {}
            pbar = tqdm_local(data_list, desc="Loading images", ncols=100, file=sys.stdout)
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
            print(
                f"✓ Loaded {len(self._preloaded_images)} images into memory "
                f"({len(self._preloaded_images) * 65 / 1024:.2f} MB)"
            )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        img_path = item['img_path']
        latex = item['latex']

        # Preprocess image (with preloading or caching)
        if self._preloaded_images is not None:
            img_tensor = self._preloaded_images.get(
                img_path, torch.zeros(1, 64, 256, dtype=torch.float32)
            )
        elif self.cache_images and img_path in self._image_cache:
            img_tensor = self._image_cache[img_path]
        else:
            try:
                img = preprocess_for_model(img_path)
                img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0)  # 1 x H x W
                if self.cache_images:
                    self._image_cache[img_path] = img_tensor
            except Exception as e:
                if idx < 10:
                    print(f"Error loading image {img_path}: {e}")
                img_tensor = torch.zeros(1, 64, 256, dtype=torch.float32)

        # Tokenization (as in train_im2latex/test_validation)
        tokens = [self.sos_token]
        tokens.extend(self.vocab.get(char, self.unk_token) for char in latex)
        tokens.append(self.eos_token)

        # Pad or truncate
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
    """Collate function for DataLoader."""
    imgs, input_seqs, targets, latex_texts = zip(*batch)
    imgs = torch.stack(imgs)
    input_seqs = torch.stack(input_seqs)
    targets = torch.stack(targets)
    return imgs, input_seqs, targets, latex_texts


def tokens_to_text(tokens, idx_to_char, eos_token=1, pad_token=2):
    """Converts sequence of tokens to text."""
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
    """Calculates token-level accuracy (as in test_validation.py)."""
    pred_clean = [t for t in pred_tokens if t != pad_token and t != eos_token]
    target_clean = [t for t in target_tokens if t != pad_token and t != eos_token]

    if len(target_clean) == 0:
        return 1.0 if len(pred_clean) == 0 else 0.0

    min_len = min(len(pred_clean), len(target_clean))
    if min_len == 0:
        return 0.0

    correct = sum(1 for i in range(min_len) if pred_clean[i] == target_clean[i])
    return correct / len(target_clean)


def greedy_decode(model, img_tensor, vocab, max_len=256, sos_token=0, eos_token=1):
    """Greedy decoder, optimized (as in test_validation.py)."""
    model.eval()
    device = next(model.parameters()).device
    img_tensor = img_tensor.to(device)

    decoded = [sos_token]

    with torch.no_grad():
        enc_feat = model.encoder(img_tensor)  # 1 x seq_len x C
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

    if decoded and decoded[0] == sos_token:
        decoded = decoded[1:]

    return decoded


def evaluate_model(model, dataloader, criterion, vocab, device, split_name="val", num_examples=5):
    """Evaluate model on dataset (loss + token-level accuracy + examples)."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    total_accuracy = 0.0
    total_samples = 0

    idx_to_char = {idx: char for char, idx in vocab.items()}
    eos_token = vocab.get('<EOS>', 1)
    pad_token = vocab.get('<PAD>', 2)

    examples = []
    example_count = 0

    non_blocking = torch.cuda.is_available()

    with torch.no_grad():
        for batch_idx, (imgs, input_seqs, targets, latex_texts) in enumerate(
            tqdm(dataloader, desc=f"Evaluating ({split_name})", ncols=100)
        ):
            imgs = imgs.to(device, non_blocking=non_blocking)
            input_seqs = input_seqs.to(device, non_blocking=non_blocking)
            targets = targets.to(device, non_blocking=non_blocking)

            logits = model(imgs, input_seqs)
            logits_flat = logits.reshape(-1, logits.size(-1))
            targets_flat = targets.reshape(-1)

            mask = (targets_flat != pad_token)
            if mask.sum() > 0:
                loss = criterion(logits_flat[mask], targets_flat[mask])
                total_loss += loss.item()
                num_batches += 1

            # token-level accuracy (fast, batched)
            pred_tokens_batch = logits.argmax(dim=-1).cpu().numpy()
            targets_batch = targets.cpu().numpy()

            batch_size = imgs.size(0)
            for i in range(batch_size):
                pred_seq = pred_tokens_batch[i]
                target_seq = targets_batch[i]

                accuracy = calculate_accuracy(pred_seq, target_seq, pad_token, eos_token)
                total_accuracy += accuracy
                total_samples += 1

                if example_count < num_examples:
                    img = imgs[i:i + 1]
                    pred_tokens = greedy_decode(
                        model,
                        img,
                        vocab,
                        max_len=256,
                        sos_token=vocab.get('<SOS>', 0),
                        eos_token=eos_token,
                    )
                    pred_text = tokens_to_text(pred_tokens, idx_to_char, eos_token, pad_token)
                    target_text = latex_texts[i]
                    examples.append(
                        {
                            "target": target_text,
                            "predicted": pred_text,
                            "accuracy": accuracy,
                        }
                    )
                    example_count += 1

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    avg_accuracy = total_accuracy / total_samples if total_samples > 0 else 0.0

    return avg_loss, avg_accuracy, examples


def build_loader(data_list, vocab, max_len, batch_size, split_name="train"):
    """Creates DataLoader for train/val with the same optimizations as in other scripts."""
    if not data_list:
        return None

    # RAM strategy - as in train_im2latex/test_validation
    if psutil is not None:
        available_ram_gb = psutil.virtual_memory().available / 1024**3
    else:
        available_ram_gb = 16

    estimated_ram_gb = (len(data_list) * 65 * 1024) / 1024**3

    print(f"\n[{split_name}] Available RAM: {available_ram_gb:.2f} GB")
    print(f"[{split_name}] Estimated RAM for preload: {estimated_ram_gb:.2f} GB")

    if available_ram_gb > estimated_ram_gb * 1.5:
        preload_images = True
        cache_images = False
        print(f"[{split_name}] ✓ Using preload of all images into RAM (fast!)")
    elif available_ram_gb > 8:
        preload_images = False
        cache_images = True
        print(f"[{split_name}] ✓ Using image caching")
    else:
        preload_images = False
        cache_images = False
        print(f"[{split_name}] ⚠ Loading from disk (requires more RAM for acceleration)")

    dataset = Im2LatexDataset(
        data_list,
        vocab,
        max_len=max_len,
        cache_images=cache_images,
        preload_images=preload_images,
    )

    use_pin_memory = torch.cuda.is_available()
    num_workers = 0  # safe and compatible with Windows, especially with preloading

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split_name == "train"),
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
    )

    print(f"[{split_name}] ✓ Dataset created, batches: {len(dataloader)}")
    return dataloader


def main():
    json_path = Path("datasets/im2latex_prepared.json")
    model_path = Path("models/im2latex_best.pth")
    vocab_path = Path("models/vocab.json")
    batch_size = 32
    max_len = 256
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_examples = 5

    print("=" * 80)
    print("EVALUATION ON TRAIN AND VALIDATION SETS")
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
        return

    print(f"Loading data from {json_path}...")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    train_data = data.get("train", [])
    val_data = data.get("val", [])

    if not train_data:
        print("❌ Error: Train set not found in JSON file!")
    if not val_data:
        print("❌ Error: Validation set (val) not found in JSON file!")
    if not train_data and not val_data:
        return

    print(f"✓ Train samples: {len(train_data)}")
    print(f"✓ Val samples:   {len(val_data)}")
    print()

    # Load/Build vocabulary
    print("Loading vocabulary...")
    if vocab_path.exists():
        vocab = build_vocab(vocab_file=str(vocab_path))
        print(f"Loaded vocabulary from {vocab_path}")
    else:
        vocab = build_vocab(json_path=str(json_path))
        print("Vocabulary built from JSON")

    vocab_size = len(vocab)
    print(f"✓ Vocabulary size: {vocab_size}")
    print()

    # DataLoaders
    train_loader = build_loader(
        train_data, vocab, max_len=max_len, batch_size=batch_size, split_name="train"
    )
    val_loader = build_loader(
        val_data, vocab, max_len=max_len, batch_size=batch_size, split_name="val"
    )

    # Load model
    print(f"\nLoading model from {model_path}...")
    model = Im2Latex(vocab_size=vocab_size).to(device)

    try:
        checkpoint = torch.load(model_path, map_location=device)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
            print(f"✓ Model loaded (epoch {checkpoint.get('epoch', '?')})")
        else:
            model.load_state_dict(checkpoint)
            print("✓ Model loaded (old state_dict format)")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    print(f"✓ Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()

    criterion = nn.CrossEntropyLoss(ignore_index=2)

    # Evaluate on train
    if train_loader is not None:
        print("-" * 80)
        print("Evaluating on TRAIN set...")
        train_loss, train_acc, train_examples = evaluate_model(
            model, train_loader, criterion, vocab, device, split_name="train", num_examples=num_examples
        )
        print("\n=== RESULTS ON TRAIN ===")
        print(f"Average Train Loss: {train_loss:.4f}")
        print(f"Average Train Accuracy (token-level): {train_acc * 100:.2f}%")

    # Evaluate on val
    if val_loader is not None:
        print("-" * 80)
        print("Evaluating on VALIDATION set...")
        val_loss, val_acc, val_examples = evaluate_model(
            model, val_loader, criterion, vocab, device, split_name="val", num_examples=num_examples
        )
        print("\n=== RESULTS ON VAL ===")
        print(f"Average Val Loss: {val_loss:.4f}")
        print(f"Average Val Accuracy (token-level): {val_acc * 100:.2f}%")

    print("\nTrain/Val evaluation completed.")


if __name__ == "__main__":
    main()


