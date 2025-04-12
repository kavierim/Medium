import torch
import torch.nn as nn

# Assuming DecoderBlock class is defined as above
# Assuming LayerNormalization class from Part 4 is defined/imported
from part4a import LayerNormalization
from part6a import DecoderBlock 

class Decoder(nn.Module):
    """
    Implements the full Transformer Decoder, consisting of a stack of DecoderBlocks.
    """
    def __init__(self, features: int, layers: nn.ModuleList):
        """
        Args:
            features (int): Number of features (d_model). Used for final LayerNorm.
            layers (nn.ModuleList): A list of pre-initialized DecoderBlock layers.
        """
        super().__init__()
        self.layers = layers
        # Final LayerNorm after the stack (consistent with Encoder implementation)
        self.norm = LayerNormalization(features)

    def forward(self, x: torch.Tensor, encoder_output: torch.Tensor, src_mask: torch.Tensor, tgt_mask: torch.Tensor):
        """
        Forward pass for the entire Decoder stack.

        Args:
            x (torch.Tensor): Input tensor (target embeddings + positional encoding),
                              shape (batch_size, tgt_seq_len, d_model).
            encoder_output (torch.Tensor): Output from the Encoder stack,
                                           shape (batch_size, src_seq_len, d_model).
            src_mask (torch.Tensor): Source mask for cross-attention layers.
            tgt_mask (torch.Tensor): Target mask for self-attention layers.

        Returns:
            torch.Tensor: Output tensor of the decoder stack, shape (batch_size, tgt_seq_len, d_model).
        """
        # Pass the input through each layer in the stack
        for layer in self.layers:
            # Pass all required inputs to each decoder block
            x = layer(x, encoder_output, src_mask, tgt_mask)
        # Apply final layer normalization
        return self.norm(x)