import torch

from part8a import *
from part8b import *
from part8c import *

def create_src_mask(src_ids: torch.Tensor, pad_id: int):
    """
    Creates a mask for source sequence padding.

    Args:
        src_ids (torch.Tensor): Source sequence IDs, shape (batch_size, src_seq_len).
        pad_id (int): The ID used for padding.

    Returns:
        torch.Tensor: Source mask, shape (batch_size, 1, 1, src_seq_len). True where padded.
    """
    # Check where the source sequence equals the padding ID
    src_mask = (src_ids == pad_id)
    # Add dimensions for heads and query sequence length for broadcasting
    # Shape becomes (batch_size, 1, 1, src_seq_len)
    return src_mask.unsqueeze(1).unsqueeze(2)

def create_tgt_mask(tgt_ids: torch.Tensor, pad_id: int):
    """
    Creates a combined mask for target sequence padding and look-ahead.

    Args:
        tgt_ids (torch.Tensor): Target sequence IDs, shape (batch_size, tgt_seq_len).
        pad_id (int): The ID used for padding.

    Returns:
        torch.Tensor: Target mask, shape (batch_size, 1, tgt_seq_len, tgt_seq_len). True where padded or future.
    """
    batch_size, tgt_seq_len = tgt_ids.shape

    # 1. Create padding mask (similar to source mask, but for target)
    # Shape: (batch_size, 1, 1, tgt_seq_len)
    tgt_padding_mask = (tgt_ids == pad_id).unsqueeze(1).unsqueeze(2)

    # 2. Create look-ahead mask
    # Shape: (tgt_seq_len, tgt_seq_len)
    look_ahead_mask = torch.triu(torch.ones(tgt_seq_len, tgt_seq_len, device=tgt_ids.device), diagonal=1).bool()
    # Shape: (1, 1, tgt_seq_len, tgt_seq_len) after unsqueezing for batch/head dims
    look_ahead_mask = look_ahead_mask.unsqueeze(0).unsqueeze(0)

    # 3. Combine padding mask and look-ahead mask
    # Padding mask shape is broadcasted to (batch_size, 1, tgt_seq_len, tgt_seq_len)
    # Look-ahead mask shape is broadcasted to (batch_size, 1, tgt_seq_len, tgt_seq_len)
    # Combine using logical OR: mask is True if EITHER padding OR look-ahead applies.
    tgt_mask = tgt_padding_mask | look_ahead_mask
    return tgt_mask

# Example mask creation
dummy_src_tensor = torch.tensor([src_padded]) # Add batch dimension
dummy_tgt_tensor = torch.tensor([tgt_padded])

src_mask_example = create_src_mask(dummy_src_tensor, PAD_ID)
tgt_mask_example = create_tgt_mask(dummy_tgt_tensor, PAD_ID)

print(f"\nSource Mask shape: {src_mask_example.shape}") # Should be [1, 1, 1, MAX_PADDED_LEN]
print(f"Target Mask shape: {tgt_mask_example.shape}") # Should be [1, 1, MAX_PADDED_LEN, MAX_PADDED_LEN]
# print("Target Mask example:\n", tgt_mask_example.squeeze()) # Print one mask