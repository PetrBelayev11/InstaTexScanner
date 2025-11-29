import json
import re
from pathlib import Path
from collections import Counter

def tokenize_latex(latex_str):
    """
    Properly tokenize LaTeX into meaningful tokens.
    """
    # Remove comments first
    latex_str = re.sub(r'%.*$', '', latex_str, flags=re.MULTILINE)
    
    # Better pattern: match in order of priority
    tokens = []
    i = 0
    n = len(latex_str)
    
    while i < n:
        # 1. LaTeX commands: \command or \command[option]{arg}
        if latex_str[i] == '\\':
            # Match LaTeX command: \command or single character command like \,
            cmd_match = re.match(r'\\([a-zA-Z]+|[^a-zA-Z])', latex_str[i:])
            if cmd_match:
                cmd = '\\' + cmd_match.group(1)
                tokens.append(cmd)
                i += len(cmd)
                continue
        
        # 2. Match environments: \begin{env} and \end{env}
        if latex_str[i:i+6] == '\\begin':
            env_match = re.match(r'\\begin\{([^}]+)\}', latex_str[i:])
            if env_match:
                tokens.append('\\begin{' + env_match.group(1) + '}')
                i += len('\\begin{' + env_match.group(1) + '}')
                continue
        
        if latex_str[i:i+4] == '\\end':
            env_match = re.match(r'\\end\{([^}]+)\}', latex_str[i:])
            if env_match:
                tokens.append('\\end{' + env_match.group(1) + '}')
                i += len('\\end{' + env_match.group(1) + '}')
                continue
        
        # 3. Special characters (individual tokens)
        if latex_str[i] in '{}[]()^_$':
            tokens.append(latex_str[i])
            i += 1
            continue
        
        # 4. Operators
        if latex_str[i] in '+-*/=<>':
            tokens.append(latex_str[i])
            i += 1
            continue
        
        # 5. Punctuation
        if latex_str[i] in ',.;!?':
            tokens.append(latex_str[i])
            i += 1
            continue
        
        # 6. Numbers (individual digits, don't combine them)
        if latex_str[i].isdigit():
            # Just take single digits, don't combine numbers
            tokens.append(latex_str[i])
            i += 1
            continue
        
        # 7. Letters (individual letters for variables)
        if latex_str[i].isalpha():
            tokens.append(latex_str[i])
            i += 1
            continue
        
        # 8. Whitespace (skip it)
        if latex_str[i].isspace():
            i += 1
            continue
        
        # 9. Anything else as individual character
        tokens.append(latex_str[i])
        i += 1
    
    return tokens

def build_latex_vocab(json_path=None, vocab_file=None, min_freq=2):
    """
    Build proper LaTeX vocabulary with meaningful tokens.
    """
    if vocab_file and Path(vocab_file).exists():
        with open(vocab_file, 'r', encoding='utf-8') as f:
            vocab = json.load(f)
        print(f"Loaded vocabulary from {vocab_file} ({len(vocab)} tokens)")
        return vocab
    
    if json_path is None:
        json_path = Path(__file__).parent.parent.parent / "datasets" / "im2latex_prepared.json"
    
    if not Path(json_path).exists():
        print(f"❌ Data file not found: {json_path}")
        return get_default_latex_vocab()
    
    print(f"Building LaTeX vocabulary from {json_path}...")
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"❌ Error loading JSON: {e}")
        return get_default_latex_vocab()
    
    # Test tokenizer first
    print("Testing tokenizer with sample LaTeX:")
    test_cases = [
        "\\frac{1}{2}",
        "x^2 + y^2 = z^2",
        "\\sum_{i=1}^{n} i = \\frac{n(n+1)}{2}",
        "123.456",
        "\\alpha\\beta\\gamma"
    ]
    
    for test in test_cases:
        tokens = tokenize_latex(test)
        print(f"  '{test}' -> {tokens}")
    
    # Collect all tokens from training LaTeX formulas
    all_tokens = []
    token_count = 0
    
    for split in ['train', 'val', 'test']:
        if split in data:
            print(f"Processing {split} split...")
            for i, item in enumerate(data[split]):
                try:
                    tokens = tokenize_latex(item['latex'])
                    all_tokens.extend(tokens)
                    token_count += 1
                    
                    # Show first few examples
                    if i < 3:
                        print(f"    Sample {i}: '{item['latex'][:50]}...' -> {tokens[:10]}...")
                        
                except Exception as e:
                    if i < 5:  # Only show first few errors
                        print(f"Warning: Error tokenizing item {i} in {split}: {e}")
                    continue
    
    print(f"Tokenized {token_count} formulas")
    
    if not all_tokens:
        print("❌ No tokens found! Using default vocabulary.")
        return get_default_latex_vocab()
    
    # Count tokens and filter by frequency
    token_counts = Counter(all_tokens)
    
    # Build vocabulary
    vocab = {
        '<SOS>': 0,
        '<EOS>': 1, 
        '<PAD>': 2,
        '<UNK>': 3,
    }
    
    # Add tokens that meet frequency threshold
    idx = len(vocab)
    added_tokens = 0
    
    # Add special tokens first
    special_tokens = ['{', '}', '(', ')', '[', ']', '^', '_', '=', '+', '-', '*', '/']
    for token in special_tokens:
        if token_counts[token] >= min_freq and token not in vocab:
            vocab[token] = idx
            idx += 1
            added_tokens += 1
    
    # Then add all other tokens by frequency
    for token, count in token_counts.most_common():
        if count >= min_freq and token not in vocab:
            vocab[token] = idx
            idx += 1
            added_tokens += 1
    
    print(f"Built LaTeX vocabulary with {len(vocab)} tokens")
    print(f"Added {added_tokens} tokens (min_freq={min_freq})")
    print(f"Most common tokens: {token_counts.most_common(30)}")
    
    if vocab_file:
        try:
            with open(vocab_file, 'w', encoding='utf-8') as f:
                json.dump(vocab, f, ensure_ascii=False, indent=2)
            print(f"✅ Saved vocabulary to {vocab_file}")
        except Exception as e:
            print(f"❌ Error saving vocabulary: {e}")
    
    return vocab

def get_default_latex_vocab():
    """Default vocabulary with common LaTeX tokens."""
    default_vocab = {
        '<SOS>': 0, '<EOS>': 1, '<PAD>': 2, '<UNK>': 3,
        # LaTeX commands
        '\\frac': 4, '\\sqrt': 5, '\\sum': 6, '\\int': 7, '\\lim': 8,
        '\\alpha': 9, '\\beta': 10, '\\gamma': 11, '\\delta': 12,
        '\\pi': 13, '\\theta': 14, '\\sigma': 15,
        '\\cdot': 16, '\\times': 17, '\\div': 18,
        '\\left': 19, '\\right': 20,
        '\\begin': 21, '\\end': 22,
        # Math symbols
        '{': 23, '}': 24, '(': 25, ')': 26, '[': 27, ']': 28,
        '^': 29, '_': 30, '=': 31, '+': 32, '-': 33, '*': 34, '/': 35,
        # Common elements
        'x': 36, 'y': 37, 'z': 38, 'a': 39, 'b': 40, 'c': 41,
        '0': 42, '1': 43, '2': 44, '3': 45, '4': 46, '5': 47, '6': 48, '7': 49, '8': 50, '9': 51,
        ',': 52, '.': 53
    }
    return default_vocab

class LatexTokenizer:
    def __init__(self, vocab):
        self.vocab = vocab
        self.unk_token = vocab.get('<UNK>', 3)
        self.idx_to_token = {v: k for k, v in vocab.items()}
    
    def encode(self, latex_str):
        tokens = tokenize_latex(latex_str)
        indices = [self.vocab.get(token, self.unk_token) for token in tokens]
        return indices
    
    def decode(self, indices):
        tokens = [self.idx_to_token.get(idx, '<UNK>') for idx in indices if idx not in [0, 1, 2]]  # Skip SOS, EOS, PAD
        return ''.join(tokens)

if __name__ == "__main__":
    # Build vocabulary
    vocab = build_latex_vocab(
        json_path="../../datasets/im2latex_prepared.json",
        vocab_file="../../models/vocab.json",  # Different filename to avoid overwriting
        min_freq=2
    )
    
    # Test the new tokenizer
    print("\n" + "="*50)
    print("Testing new tokenizer:")
    
    tokenizer = LatexTokenizer(vocab)
    