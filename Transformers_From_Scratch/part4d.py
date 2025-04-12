import torch
import torch.nn as nn

# assuming the supporting modules are defined in separate files
from part4a import LayerNormalization
from part4b import PositionwiseFeedForward
from part4c import ResidualConnection

# --- Unit Testing / Verification ---
print("\n--- Testing Supporting Modules ---")

# Parameters
batch_size_test = 4
seq_len_test = 10
d_model_test = 512
d_ff_test = 2048 # Typical inner FFN dimension
dropout_test = 0.1

# Dummy input
dummy_input = torch.randn(batch_size_test, seq_len_test, d_model_test)
print(f"Dummy Input shape: {dummy_input.shape}")

# --- Test LayerNormalization ---
ln_layer = LayerNormalization(d_model_test)
ln_output = ln_layer(dummy_input)
print(f"Output shape after LayerNorm: {ln_output.shape}") # Expected: [4, 10, 512]
assert ln_output.shape == dummy_input.shape
# Check if mean is close to 0 and std is close to 1 after norm (before scale/shift)
# Note: We test the internal logic conceptually here; exact 0/1 depends on eps and learned params
print("LayerNormalization test passed (shape check).")

# --- Test PositionwiseFeedForward ---
ffn_layer = PositionwiseFeedForward(d_model_test, d_ff_test, dropout_test)
ffn_output = ffn_layer(dummy_input)
print(f"Output shape after FFN: {ffn_output.shape}") # Expected: [4, 10, 512]
assert ffn_output.shape == dummy_input.shape
print("PositionwiseFeedForward test passed (shape check).")

# --- Test ResidualConnection ---
# Create a dummy sublayer (e.g., a simple Linear layer for shape testing)
dummy_sublayer = nn.Linear(d_model_test, d_model_test)
residual_layer = ResidualConnection(d_model_test, dropout_test)
residual_output = residual_layer(dummy_input, dummy_sublayer) # Pass input and sublayer
print(f"Output shape after ResidualConnection: {residual_output.shape}") # Expected: [4, 10, 512]
assert residual_output.shape == dummy_input.shape
print("ResidualConnection test passed (shape check).")