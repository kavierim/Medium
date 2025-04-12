import torch
import torch.nn as nn

# Assuming LayerNormalization class from Part 4 is defined/imported
from part4b import PositionwiseFeedForward
from part5a import MultiHeadAttention
from part6a import DecoderBlock
from part6b import Decoder

# --- Unit Testing / Verification ---
print("\n--- Testing Decoder ---")

# Parameters
batch_size_test = 4
src_seq_len_test = 12 # Source sequence length
tgt_seq_len_test = 10 # Target sequence length
d_model_test = 512
num_heads_test = 8
d_ff_test = 2048
num_layers_test = 6 # N=6 as in the paper
dropout_test = 0.1

# Dummy inputs
dummy_tgt = torch.randn(batch_size_test, tgt_seq_len_test, d_model_test) # Target input
dummy_encoder_output = torch.randn(batch_size_test, src_seq_len_test, d_model_test) # From Encoder

# Dummy masks
# Source mask (padding mask for encoder output) -> shape (batch, 1, 1, src_seq_len)
dummy_src_mask_dec = torch.zeros(batch_size_test, 1, 1, src_seq_len_test, dtype=torch.bool)
if batch_size_test >= 1: dummy_src_mask_dec[0, :, :, -3:] = True # Example padding
if batch_size_test >= 2: dummy_src_mask_dec[1, :, :, -2:] = True

# Target mask (look-ahead + padding) -> shape (batch, 1, tgt_seq_len, tgt_seq_len)
# Combine look-ahead and potential padding mask for target
tgt_padding_mask = torch.zeros(batch_size_test, 1, 1, tgt_seq_len_test, dtype=torch.bool) # Example padding mask
if batch_size_test >= 1: tgt_padding_mask[0, :, :, -1:] = True
look_ahead_mask = torch.triu(torch.ones(tgt_seq_len_test, tgt_seq_len_test), diagonal=1).bool().unsqueeze(0).unsqueeze(0) # Shape [1, 1, T, T]
# Combine: mask is True if EITHER look-ahead OR padding applies
dummy_tgt_mask_dec = (tgt_padding_mask | look_ahead_mask) # Logical OR, shape broadcasts to [B, 1, T, T]

print(f"Dummy Target Input shape: {dummy_tgt.shape}")
print(f"Dummy Encoder Output shape: {dummy_encoder_output.shape}")
print(f"Dummy Source Mask shape: {dummy_src_mask_dec.shape}")
print(f"Dummy Target Mask shape: {dummy_tgt_mask_dec.shape}")

# Build Decoder components
self_attention = MultiHeadAttention(d_model_test, num_heads_test, dropout_test)
cross_attention = MultiHeadAttention(d_model_test, num_heads_test, dropout_test)
feed_forward = PositionwiseFeedForward(d_model_test, d_ff_test, dropout_test)

# Create a stack of DecoderBlocks
decoder_layers = nn.ModuleList([
    DecoderBlock(d_model_test, self_attention, cross_attention, feed_forward, dropout_test)
    for _ in range(num_layers_test)
])

# Instantiate the Decoder
decoder = Decoder(d_model_test, decoder_layers)

# Perform forward pass
decoder_output = decoder(dummy_tgt, dummy_encoder_output, dummy_src_mask_dec, dummy_tgt_mask_dec)

# Check output shape
print(f"Output shape after Decoder: {decoder_output.shape}") # Expected: [4, 10, 512]
assert decoder_output.shape == dummy_tgt.shape # Output seq len should match target input seq len

print("Decoder test passed (shape check).")