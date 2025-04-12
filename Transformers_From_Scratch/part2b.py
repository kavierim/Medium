import torch

# Import the scaled_dot_product_attention function from part2a
from part2a import scaled_dot_product_attention

# --- Unit Testing / Verification ---
print("\n--- Testing Scaled Dot-Product Attention ---")

# Parameters (Example)
batch_size_test = 2
num_heads_test = 8 # Number of attention heads (will be relevant in Part 3)
seq_len_q_test = 5 # Sequence length for queries
seq_len_k_test = 7 # Sequence length for keys/values
d_k_test = 64 // num_heads_test # Dimension of keys/queries per head
d_v_test = 64 // num_heads_test # Dimension of values per head
d_model_test = 64 # Total model dimension (d_k * num_heads)

# Dummy tensors
# Note the shape: (batch_size, num_heads, seq_len, dim_per_head)
dummy_q = torch.randn(batch_size_test, num_heads_test, seq_len_q_test, d_k_test)
dummy_k = torch.randn(batch_size_test, num_heads_test, seq_len_k_test, d_k_test)
dummy_v = torch.randn(batch_size_test, num_heads_test, seq_len_k_test, d_v_test) # seq_len_k == seq_len_v

print(f"Dummy Q shape: {dummy_q.shape}")
print(f"Dummy K shape: {dummy_k.shape}")
print(f"Dummy V shape: {dummy_v.shape}")

# --- Test without mask ---
print("\nTesting without mask:")
output_no_mask, weights_no_mask = scaled_dot_product_attention(dummy_q, dummy_k, dummy_v)
print(f"Output shape (no mask): {output_no_mask.shape}")       # Expected: [2, 8, 5, 8] (batch, heads, seq_q, dv)
print(f"Weights shape (no mask): {weights_no_mask.shape}")     # Expected: [2, 8, 5, 7] (batch, heads, seq_q, seq_k)
assert output_no_mask.shape == (batch_size_test, num_heads_test, seq_len_q_test, d_v_test)
assert weights_no_mask.shape == (batch_size_test, num_heads_test, seq_len_q_test, seq_len_k_test)
# Check if weights sum to 1 (approximately due to float precision)
assert torch.allclose(weights_no_mask.sum(dim=-1), torch.ones_like(weights_no_mask.sum(dim=-1))), "Weights don't sum to 1"
print("Test without mask passed.")

# --- Test with a simple look-ahead mask (for decoder self-attention) ---
# Create a mask for the query sequence length (seq_len_q_test = 5)
# Mask should be (seq_len_q, seq_len_k). Here seq_q=seq_k=5 for self-attention mask
mask_size = seq_len_q_test
# Create an upper triangular matrix (True above the diagonal)
look_ahead_mask = torch.triu(torch.ones(mask_size, mask_size), diagonal=1).bool()
# Expand mask to match attention score shape (batch_size, num_heads, seq_len_q, seq_len_k)
# We can use broadcasting: Unsqueeze for batch and head dimensions.
# Mask shape needs to be compatible: (1, 1, seq_len_q, seq_len_k) or similar
dummy_mask = look_ahead_mask.unsqueeze(0).unsqueeze(0) # Shape: [1, 1, 5, 5]

# Adjust dummy K/V seq len to match Q seq len for self-attention mask test
dummy_k_masked = torch.randn(batch_size_test, num_heads_test, seq_len_q_test, d_k_test)
dummy_v_masked = torch.randn(batch_size_test, num_heads_test, seq_len_q_test, d_v_test)

print("\nTesting with look-ahead mask:")
print(f"Look-ahead mask shape: {dummy_mask.shape}")
print("Mask (True means masked):")
print(look_ahead_mask) # Show the basic mask structure

output_masked, weights_masked = scaled_dot_product_attention(dummy_q, dummy_k_masked, dummy_v_masked, mask=dummy_mask)
print(f"Output shape (masked): {output_masked.shape}")   # Expected: [2, 8, 5, 8]
print(f"Weights shape (masked): {weights_masked.shape}") # Expected: [2, 8, 5, 5]
assert output_masked.shape == (batch_size_test, num_heads_test, seq_len_q_test, d_v_test)
assert weights_masked.shape == (batch_size_test, num_heads_test, seq_len_q_test, seq_len_q_test) # seq_k adjusted
# Check if masked weights are zero
masked_positions_weights = weights_masked.masked_select(dummy_mask)
assert torch.all(masked_positions_weights == 0), "Masked positions have non-zero weights!"
# Check if non-masked weights sum to 1
# Note: Need to be careful here, sum across seq_k dim
assert torch.allclose(weights_masked.sum(dim=-1), torch.ones_like(weights_masked.sum(dim=-1))), "Weights don't sum to 1"

print("Test with mask passed.")