import torch
import torch.nn as nn

# Assume Transformer class and all sub-modules are defined/imported
# Assume create_src_mask, create_tgt_mask functions are defined/imported

from part8g import *


def greedy_decode(model: Transformer, src: torch.Tensor, src_mask: torch.Tensor, max_len: int, sos_id: int, eos_id: int, device: torch.device):
    """
    Performs greedy decoding to generate an output sequence.

    Args:
        model (Transformer): The trained Transformer model.
        src (torch.Tensor): Source sequence IDs, shape (1, src_seq_len). Batch size must be 1.
        src_mask (torch.Tensor): Source mask, shape (1, 1, 1, src_seq_len).
        max_len (int): Maximum length of the generated sequence.
        sos_id (int): Token ID for the Start-Of-Sequence token.
        eos_id (int): Token ID for the End-Of-Sequence token.
        device (torch.device): Device to run inference on (e.g., 'cuda' or 'cpu').

    Returns:
        torch.Tensor: The generated sequence of token IDs, shape (1, generated_seq_len).
    """
    model.eval() # Set model to evaluation mode
    src = src.to(device)
    src_mask = src_mask.to(device)

    with torch.no_grad(): # Disable gradient calculations
        # 1. Encode the source sequence
        encoder_output = model.encode(src, src_mask) # Shape: (1, src_seq_len, d_model)

        # 2. Initialize decoder input with SOS token
        # Shape: (1, 1) representing [SOS]
        decoder_input = torch.tensor([[sos_id]], dtype=torch.long, device=device)

        # 3. Loop until EOS or max_len
        for _ in range(max_len - 1): # -1 because we already have SOS
            # 3a. Create target mask for the current decoder input sequence
            # Shape: (1, 1, current_tgt_len, current_tgt_len)
            tgt_mask = create_tgt_mask(decoder_input, PAD_ID).to(device) # Assuming PAD_ID is accessible

            # 3b. Decode using encoder output and current decoder input
            # Shape: (1, current_tgt_len, d_model)
            decoder_output = model.decode(encoder_output, src_mask, decoder_input, tgt_mask)

            # 3c. Project the *last* decoder output token to get logits
            # Shape: (1, 1, vocab_size)
            logits = model.project(decoder_output[:, -1:]) # Select only the last time step output

            # 3d. Select the token ID with the highest logit (greedy choice)
            # Shape: (1, 1)
            _, next_token_id = torch.max(logits, dim=-1)

            # 3e. Append the predicted token ID to the decoder input sequence
            # Shape becomes (1, current_tgt_len + 1)
            decoder_input = torch.cat([decoder_input, next_token_id], dim=1)

            # 3f. Check if EOS token was generated
            if next_token_id.item() == eos_id:
                break

    # Return the generated sequence (excluding the initial SOS token if desired)
    # return decoder_input[:, 1:] # Exclude SOS
    return decoder_input # Include SOS for clarity maybe