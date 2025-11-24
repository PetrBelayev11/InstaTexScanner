import json
from pathlib import Path
from collections import Counter


def build_vocab(json_path: str = None, vocab_file: str = None):
    """
    Build vocabulary from training data or load from file.
    
    Args:
        json_path: Path to training JSON file (if building new vocab)
        vocab_file: Path to save/load vocabulary file
    
    Returns:
        Dictionary mapping characters to indices
    """
    if vocab_file and Path(vocab_file).exists():
        # Load existing vocabulary
        with open(vocab_file, 'r', encoding='utf-8') as f:
            vocab = json.load(f)
        print(f"Loaded vocabulary from {vocab_file} ({len(vocab)} tokens)")
        return vocab
    
    if json_path is None:
        # Default path
        json_path = Path(__file__).parent.parent / "datasets" / "im2latex_prepared.json"
    
    if not Path(json_path).exists():
        # Return default minimal vocabulary if file doesn't exist
        print(f"Warning: {json_path} not found. Using default vocabulary.")
        default_vocab = {
            '<SOS>': 0, '<EOS>': 1, '<PAD>': 2, '<UNK>': 3
        }
        # Add common LaTeX characters
        chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789+-=()[]{}\\^_"
        for i, char in enumerate(chars):
            default_vocab[char] = i + 4
        return default_vocab
    
    # Build vocabulary from training data
    print(f"Building vocabulary from {json_path}...")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Collect all characters from training LaTeX formulas
    all_chars = []
    for split in ['train', 'val', 'test']:
        if split in data:
            for item in data[split]:
                all_chars.extend(list(item['latex']))
    
    # Count characters
    char_counts = Counter(all_chars)
    
    # Build vocabulary: special tokens + all unique characters
    vocab = {
        '<SOS>': 0,  # Start of sequence
        '<EOS>': 1,  # End of sequence
        '<PAD>': 2,  # Padding
        '<UNK>': 3,  # Unknown
    }
    
    # Add characters sorted by frequency (most common first)
    idx = len(vocab)
    for char, count in char_counts.most_common():
        if char not in vocab:
            vocab[char] = idx
            idx += 1
    
    print(f"Built vocabulary with {len(vocab)} tokens")
    
    # Save vocabulary if path provided
    if vocab_file:
        with open(vocab_file, 'w', encoding='utf-8') as f:
            json.dump(vocab, f, ensure_ascii=False, indent=2)
        print(f"Saved vocabulary to {vocab_file}")
    
    return vocab
