import torch
import torch.nn as nn
import numpy as np

print(f"PyTorch Version: {torch.__version__}")

# --- Tensor Basics ---
print("\n--- Tensor Basics ---")
# Create tensors
x = torch.tensor([[1., 2.], [3., 4.]])
y = torch.rand(2, 2) # Random tensor
print("Tensor x:\n", x)
print("Tensor y:\n", y)

# Tensor operations (Example: Matrix Multiplication)
z = torch.matmul(x, y)
print("Matrix Multiplication (x @ y):\n", z)

# Indexing
print("Element at [0, 1] of z:", z[0, 1].item()) # .item() gets Python number from scalar tensor

# Tensor shapes
print("Shape of x:", x.shape)
print("Shape of z:", z.shape)

# Reshaping (view) - often needed in Transformers
# Example: Flatten the tensor z
flat_z = z.view(-1) # -1 infers the size
print("Flattened z:", flat_z)
print("Shape of flattened z:", flat_z.shape)
# Example: Reshape back (if possible)
reshaped_z = flat_z.view(2, 2)
print("Reshaped z:\n", reshaped_z)

# --- nn.Module Basics ---
print("\n--- nn.Module Basics ---")
# Simple linear layer example
d_in = 4 # Input dimension
d_out = 8 # Output dimension
linear_layer = nn.Linear(d_in, d_out)
print("Linear Layer:", linear_layer)

# Create dummy input batch (Batch Size=3, Input Dim=4)
dummy_input = torch.randn(3, d_in)
print("Dummy Input shape:", dummy_input.shape)

# Forward pass
output = linear_layer(dummy_input)
print("Output shape:", output.shape) # Should be (3, 8)

# --- Autograd Basics ---
print("\n--- Autograd Basics ---")
# Tensors track gradients if requires_grad=True
a = torch.tensor([[1., 2.], [3., 4.]], requires_grad=True)
b = torch.tensor([[5., 6.], [7., 8.]], requires_grad=True)
c = a * b
d = c.mean() # Calculate a scalar value

print("Tensor a:", a)
print("Tensor d (scalar output):", d)

# Calculate gradients
d.backward()

# Gradients are accumulated in .grad attribute
print("Gradient of d w.r.t. a:\n", a.grad)
print("Gradient of d w.r.t. b:\n", b.grad)
