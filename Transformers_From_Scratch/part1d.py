import torch

# import earlier code
from part1a import InputEmbeddings
from part1b import PositionalEncoding

# --- Unit Testing / Verification ---
print("\n--- Testing Modules ---")

# Parameters
vocab_size_test = 1000  # Example vocabulary size
d_model_test = 512    # Must match d_model used in PositionalEncoding
max_len_test = 100     # Max sequence length for PE
dropout_test = 0.1

# Dummy input (batch_size=2, sequence_length=10)
batch_size_test = 2
seq_len_test = 10
dummy_token_ids = torch.randint(0, vocab_size_test, (batch_size_test, seq_len_test))
print(f"Dummy Token IDs shape: {dummy_token_ids.shape}")

# Instantiate modules
embedding_layer = InputEmbeddings(d_model_test, vocab_size_test)
positional_encoding_layer = PositionalEncoding(d_model_test, max_len_test, dropout_test)

# --- Test InputEmbeddings ---
embedding_output = embedding_layer(dummy_token_ids)
print(f"Output shape after InputEmbeddings: {embedding_output.shape}") # Expected: [2, 10, 512]
assert embedding_output.shape == (batch_size_test, seq_len_test, d_model_test)
print("InputEmbeddings test passed.")

# --- Test PositionalEncoding ---
# Note: PE layer takes embeddings as input
final_output = positional_encoding_layer(embedding_output)
print(f"Output shape after PositionalEncoding: {final_output.shape}") # Expected: [2, 10, 512]
assert final_output.shape == (batch_size_test, seq_len_test, d_model_test)
print("PositionalEncoding test passed.")

# Check if positional encodings were added (output should not be same as input)
assert not torch.equal(embedding_output, final_output), "Positional encoding was not added!"
print("Output differs from input after PE (as expected).")