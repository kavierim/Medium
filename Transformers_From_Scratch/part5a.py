import torch
import torch.nn as nn

# Assume MultiHeadAttention, PositionwiseFeedForward, LayerNormalization,
# and ResidualConnection classes from previous parts are defined/imported.
# Example imports (adjust paths if needed):
# from part3 import MultiHeadAttention
# from part4 import PositionwiseFeedForward, ResidualConnection
from part4b import PositionwiseFeedForward
from part4c import ResidualConnection
from part3a import MultiHeadAttention

class EncoderBlock(nn.Module):
    """
    Implements a single block of the Transformer Encoder.
    Consists of Multi-Head Self-Attention and Position-wise Feed-Forward sub-layers,
    each followed by a residual connection and layer normalization.
    """
    def __init__(self, features: int, self_attention_block: MultiHeadAttention, feed_forward_block: PositionwiseFeedForward, dropout: float):
        """
        Args:
            features (int): Number of features (d_model).
            self_attention_block (MultiHeadAttention): Pre-initialized Multi-Head Self-Attention module.
            feed_forward_block (PositionwiseFeedForward): Pre-initialized Position-wise Feed-Forward module.
            dropout (float): Dropout probability for the residual connections.
        """
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        # Create two residual connection wrappers
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(2)])

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor):
        """
        Forward pass for the EncoderBlock.

        Args:
            x (torch.Tensor): Input tensor, shape (batch_size, seq_len, d_model).
            src_mask (torch.Tensor): Source mask for the self-attention layer to hide padding tokens.
                                     Shape needs to be broadcastable to
                                     (batch_size, num_heads, seq_len, seq_len). Example: (batch_size, 1, 1, seq_len).

        Returns:
            torch.Tensor: Output tensor, shape (batch_size, seq_len, d_model).
        """
        # 1. Apply Multi-Head Self-Attention with Add & Norm
        # The sublayer function needs to accept only 'x'. We use a lambda to pass Q, K, V and the mask.
        # Note: For self-attention, query, key, and value are all the same input 'x'.
        x = self.residual_connections[0](x, lambda x_res: self.self_attention_block(x_res, x_res, x_res, src_mask))

        # 2. Apply Position-wise Feed-Forward Network with Add & Norm
        x = self.residual_connections[1](x, self.feed_forward_block)

        return x