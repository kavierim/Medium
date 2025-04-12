import torch
import torch.nn as nn

# Assuming EncoderBlock class is defined as above
# Assuming LayerNormalization class from Part 4 is defined/imported
from part4a import LayerNormalization
from part5a import EncoderBlock

class Encoder(nn.Module):
    """
    Implements the full Transformer Encoder, consisting of a stack of EncoderBlocks.
    """
    def __init__(self, features: int, layers: nn.ModuleList):
        """
        Args:
            features (int): Number of features (d_model). Used for final LayerNorm.
            layers (nn.ModuleList): A list of pre-initialized EncoderBlock layers.
        """
        super().__init__()
        self.layers = layers
        # Include final LayerNorm after the stack? Paper figure doesn't show it explicitly AFTER Nx,
        # but text implies norm is within each layer's residual connection.
        # Some implementations add it here for extra stability. Let's add it.
        self.norm = LayerNormalization(features)

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        """
        Forward pass for the entire Encoder stack.

        Args:
            x (torch.Tensor): Input tensor (from embeddings + positional encoding),
                              shape (batch_size, seq_len, d_model).
            mask (torch.Tensor): Source mask applied in all self-attention layers.

        Returns:
            torch.Tensor: Output tensor of the encoder stack, shape (batch_size, seq_len, d_model).
        """
        # Pass the input through each layer in the stack
        for layer in self.layers:
            x = layer(x, mask)
        # Apply final layer normalization
        return self.norm(x)