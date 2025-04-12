import torch
import torch.nn as nn
import math

# Assume scaled_dot_product_attention function from Part 2 is available/imported
# def scaled_dot_product_attention(query, key, value, mask=None, dropout=None): ...
from part2a import scaled_dot_product_attention

class MultiHeadAttention(nn.Module):
    """
    Implements Multi-Head Attention mechanism.
    """
    def __init__(self, d_model: int, num_heads: int, dropout: float):
        """
        Args:
            d_model (int): Total dimension of the model.
            num_heads (int): Number of attention heads.
            dropout (float): Dropout probability.
        """
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads # Dimension of keys/queries per head

        # Linear layers for Q, K, V projections (applied before splitting heads)
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        # Final linear layer after concatenating heads
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

        # We store the attention weights from the last forward pass
        # for visualization or analysis (optional)
        self.attention_weights = None

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor = None):
        """
        Forward pass for Multi-Head Attention.

        Args:
            query (torch.Tensor): Query tensor, shape (batch_size, seq_len_q, d_model)
            key (torch.Tensor): Key tensor, shape (batch_size, seq_len_k, d_model)
            value (torch.Tensor): Value tensor, shape (batch_size, seq_len_v, d_model)
                                  Note: seq_len_k and seq_len_v are typically the same.
            mask (torch.Tensor, optional): Mask tensor, typically for padding or look-ahead.
                                           Shape needs to be broadcastable for attention scores.
                                           Example shapes: (batch_size, 1, 1, seq_len_k) for padding mask,
                                                         (1, 1, seq_len_q, seq_len_k) for look-ahead mask.

        Returns:
            torch.Tensor: Output tensor, shape (batch_size, seq_len_q, d_model)
        """
        batch_size = query.shape[0]

        # 1. Apply linear projections W_q, W_k, W_v
        # Input shapes: (batch_size, seq_len, d_model)
        # Output shapes: (batch_size, seq_len, d_model)
        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)

        # 2. Reshape Q, K, V for multi-head processing
        # Change shape from (batch_size, seq_len, d_model) to
        # (batch_size, seq_len, num_heads, d_k) and then transpose to
        # (batch_size, num_heads, seq_len, d_k)
        # d_k = d_model // num_heads
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2) # Assuming d_v = d_k

        # 3. Apply Scaled Dot-Product Attention for each head
        # Q, K, V shapes: (batch_size, num_heads, seq_len, d_k)
        # Mask shape needs to be broadcastable to (batch_size, num_heads, seq_len_q, seq_len_k)
        # output shape: (batch_size, num_heads, seq_len_q, d_k)
        # attention_weights shape: (batch_size, num_heads, seq_len_q, seq_len_k)
        output, self.attention_weights = scaled_dot_product_attention(Q, K, V, mask, self.dropout)

        # 4. Concatenate heads and reshape
        # Transpose output back to (batch_size, seq_len_q, num_heads, d_k)
        output = output.transpose(1, 2).contiguous()
        # Reshape to (batch_size, seq_len_q, d_model) by combining heads
        output = output.view(batch_size, -1, self.d_model) # self.d_model = num_heads * d_k

        # 5. Apply final linear projection W_o
        # Input shape: (batch_size, seq_len_q, d_model)
        # Output shape: (batch_size, seq_len_q, d_model)
        output = self.W_o(output)

        return output