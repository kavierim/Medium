import torch
import torch.nn as nn

# Assuming the supporting modules are defined in separate files
from part5a import EncoderBlock
from part5b import Encoder
from part3a import MultiHeadAttention
from part4b import PositionwiseFeedForward


# --- Unit Testing / Verification ---
print("\n--- Testing Encoder ---")

# Parameters
batch_size_test = 4
seq_len_test = 12
d_model_test = 512
num_heads_test = 8
d_ff_test = 2048
num_layers_test = 6 # N=6 as in the paper
dropout_test = 0.1

# Dummy input (output from Embedding + Positional Encoding)
dummy_src = torch.randn(batch_size_test, seq_len_test, d_model_test)

# Dummy source mask (e.g., padding mask)
# Shape: (batch_size, 1, 1, seq_len) -> True where padded
dummy_src_mask = torch.zeros(batch_size_test, 1, 1, seq_len_test, dtype=torch.bool)
# Mask last 3 positions for first item, last 2 for second
if batch_size_test >= 1:
    dummy_src_mask[0, :, :, -3:] = True
if batch_size_test >= 2:
    dummy_src_mask[1, :, :, -2:] = True
# ... add more if needed

print(f"Dummy Source Input shape: {dummy_src.shape}")
print(f"Dummy Source Mask shape: {dummy_src_mask.shape}")

# Build Encoder components
self_attention = MultiHeadAttention(d_model_test, num_heads_test, dropout_test)
feed_forward = PositionwiseFeedForward(d_model_test, d_ff_test, dropout_test)

# Create a stack of EncoderBlocks
encoder_layers = nn.ModuleList([
    EncoderBlock(d_model_test, self_attention, feed_forward, dropout_test)
    for _ in range(num_layers_test)
])

# Instantiate the Encoder
encoder = Encoder(d_model_test, encoder_layers)

# Perform forward pass
encoder_output = encoder(dummy_src, dummy_src_mask)

# Check output shape
print(f"Output shape after Encoder: {encoder_output.shape}") # Expected: [4, 12, 512]
assert encoder_output.shape == dummy_src.shape

print("Encoder test passed (shape check).")