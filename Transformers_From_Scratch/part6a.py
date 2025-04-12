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

class DecoderBlock(nn.Module):
    """
    Implements a single block of the Transformer Decoder.
    Consists of Masked Self-Attention, Cross-Attention (Encoder-Decoder),
    and Feed-Forward sub-layers, each followed by Add & Norm.
    """
    def __init__(self, features: int, self_attention_block: MultiHeadAttention, cross_attention_block: MultiHeadAttention, feed_forward_block: PositionwiseFeedForward, dropout: float):
        """
        Args:
            features (int): Number of features (d_model).
            self_attention_block (MultiHeadAttention): Pre-initialized Masked Multi-Head Self-Attention module.
            cross_attention_block (MultiHeadAttention): Pre-initialized Multi-Head Encoder-Decoder Attention module.
            feed_forward_block (PositionwiseFeedForward): Pre-initialized Position-wise Feed-Forward module.
            dropout (float): Dropout probability for the residual connections.
        """
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        # Create three residual connection wrappers
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(3)])

    def forward(self, x: torch.Tensor, encoder_output: torch.Tensor, src_mask: torch.Tensor, tgt_mask: torch.Tensor):
        """
        Forward pass for the DecoderBlock.

        Args:
            x (torch.Tensor): Input tensor from the previous decoder layer or target embeddings,
                              shape (batch_size, tgt_seq_len, d_model).
            encoder_output (torch.Tensor): Output tensor from the Encoder stack,
                                           shape (batch_size, src_seq_len, d_model).
            src_mask (torch.Tensor): Source mask (for padding) used in the cross-attention layer.
                                     Shape needs to be broadcastable to
                                     (batch_size, num_heads, tgt_seq_len, src_seq_len). Example: (batch_size, 1, 1, src_seq_len).
            tgt_mask (torch.Tensor): Target mask (look-ahead + padding) used in the self-attention layer.
                                     Shape needs to be broadcastable to
                                     (batch_size, num_heads, tgt_seq_len, tgt_seq_len). Example: (batch_size, 1, tgt_seq_len, tgt_seq_len).

        Returns:
            torch.Tensor: Output tensor, shape (batch_size, tgt_seq_len, d_model).
        """
        # 1. Apply Masked Multi-Head Self-Attention with Add & Norm
        # Query, Key, Value are all 'x'. Use target mask (tgt_mask).
        x = self.residual_connections[0](x, lambda x_res: self.self_attention_block(x_res, x_res, x_res, tgt_mask))

        # 2. Apply Multi-Head Cross-Attention (Encoder-Decoder) with Add & Norm
        # Query is the output from the previous step ('x'). Key and Value are from the encoder_output.
        # Use source mask (src_mask).
        x = self.residual_connections[1](x, lambda x_res: self.cross_attention_block(x_res, encoder_output, encoder_output, src_mask))

        # 3. Apply Position-wise Feed-Forward Network with Add & Norm
        x = self.residual_connections[2](x, self.feed_forward_block)

        return x