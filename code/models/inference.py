import torch
from .model_im2markup import Im2Latex


def greedy_decode(model: Im2Latex, img_tensor: torch.Tensor, vocab: dict, max_len: int = 256, 
                  sos_token: int = 0, eos_token: int = 1):
    """
    Greedy decoding for im2latex model.
    
    Args:
        model: Trained Im2Latex model
        img_tensor: Preprocessed image tensor (1 x 1 x H x W)
        vocab: Vocabulary dictionary mapping characters to indices
        max_len: Maximum sequence length
        sos_token: Start of sequence token index
        eos_token: End of sequence token index
    
    Returns:
        List of token indices
    """
    model.eval()
    device = next(model.parameters()).device
    img_tensor = img_tensor.to(device)
    
    # Encode image
    with torch.no_grad():
        enc_feat = model.encoder(img_tensor)  # 1 x seq_len x C
        
        # Initialize decoder
        decoded = [sos_token]
        hidden = None
        
        for _ in range(max_len):
            # Get last token
            last_token = torch.tensor([[decoded[-1]]], device=device)
            
            # Decode one step (using encoder features as context)
            # For simplicity, we use mean pooling of encoder features
            # In a full attention model, this would be attention-weighted
            context = enc_feat.mean(dim=1, keepdim=True)  # 1 x 1 x C
            
            # Embedding
            emb = model.decoder.embedding(last_token)  # 1 x 1 x H
            
            # Combine with context (simple concatenation or addition)
            # For now, just use embedding
            out, hidden = model.decoder.gru(emb, hidden)  # 1 x 1 x H
            logits = model.decoder.fc(out)  # 1 x 1 x vocab
            
            # Greedy selection
            next_token = logits.argmax(dim=-1).item()
            decoded.append(next_token)
            
            # Stop if EOS token
            if next_token == eos_token:
                break
        
        # Remove SOS token from output
        return decoded[1:] if decoded[0] == sos_token else decoded
