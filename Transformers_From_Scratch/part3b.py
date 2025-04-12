import torch

# import earlier code
from part3a import MultiHeadAttention

# --- Unit Testing / Verification ---
print("\n--- Testing Multi-Head Attention ---")

# Parameters
batch_size_test = 4
seq_len_test = 10 # Sequence length for Q, K, V (can differ for K/V if not self-attention)
d_model_test = 512
num_heads_test = 8
dropout_test = 0.1

# Create dummy input tensors
dummy_q = torch.randn(batch_size_test, seq_len_test, d_model_test)
dummy_k = torch.randn(batch_size_test, seq_len_test, d_model_test)
dummy_v = torch.randn(batch_size_test, seq_len_test, d_model_test)

# Create a dummy mask (e.g., padding mask for key/value sequence)
# Mask shape (batch_size, 1, 1, seq_len_k) -> True where padded
dummy_mask_mha = torch.zeros(batch_size_test, 1, 1, seq_len_test, dtype=torch.bool)
# Let's mask the last 2 positions for the first batch item, last 1 for the second
if batch_size_test >= 1:
    dummy_mask_mha[0, :, :, -2:] = True
if batch_size_test >= 2:
    dummy_mask_mha[1, :, :, -1:] = True
# Add more mask examples if batch_size_test > 2...

print(f"Dummy Q/K/V shape: {dummy_q.shape}") # Expected: [4, 10, 512]
print(f"Dummy Mask shape: {dummy_mask_mha.shape}") # Expected: [4, 1, 1, 10]

# Instantiate the module
mha_layer = MultiHeadAttention(d_model_test, num_heads_test, dropout_test)

# Perform forward pass
output_mha = mha_layer(dummy_q, dummy_k, dummy_v, mask=dummy_mask_mha)

# Check output shape
print(f"Output shape after MultiHeadAttention: {output_mha.shape}") # Expected: [4, 10, 512]
assert output_mha.shape == (batch_size_test, seq_len_test, d_model_test)

# Check stored attention weights shape (optional)
# Stored weights are from the scaled_dot_product_attention call
# Shape: (batch_size, num_heads, seq_len_q, seq_len_k)
print(f"Stored Attention Weights shape: {mha_layer.attention_weights.shape}") # Expected: [4, 8, 10, 10]
assert mha_layer.attention_weights.shape == (batch_size_test, num_heads_test, seq_len_test, seq_len_test)

print("MultiHeadAttention test passed.")