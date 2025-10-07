"""
model_stub.py

A minimal PyTorch-like model stub (no heavy training here).
This file shows where to implement the im2markup / encoder-decoder model.
"""
# Example placeholder content
def build_model(vocab_size=100, embed_dim=128, hidden_dim=256):
    # Pseudo-code / interface for the model
    # Replace with actual PyTorch implementation when developing
    model = {
        "vocab_size": vocab_size,
        "embed_dim": embed_dim,
        "hidden_dim": hidden_dim
    }
    return model

if __name__ == "__main__":
    m = build_model()
    print("Model stub created:", m)
