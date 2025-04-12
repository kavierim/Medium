import torch
import torch.nn as nn
import math

def scaled_dot_product_attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor = None, dropout: nn.Dropout = None):
    """
    Computes the Scaled Dot-Product Attention.

    Args:
        query (torch.Tensor): Query tensor, shape (batch_size, num_heads, seq_len_q, d_k)
        key (torch.Tensor): Key tensor, shape (batch_size, num_heads, seq_len_k, d_k)
        value (torch.Tensor): Value tensor, shape (batch_size, num_heads, seq_len_v, d_v)
                                Note: seq_len_k and seq_len_v are typically the same.
        mask (torch.Tensor, optional): Mask tensor, shape broadcastable to
                                       (batch_size, num_heads, seq_len_q, seq_len_k).
                                       Positions with True or 1 will be masked (set to -inf). Defaults to None.
        dropout (nn.Dropout, optional): Dropout layer to apply to attention weights. Defaults to None.

    Returns:
        torch.Tensor: The output tensor after attention, shape (batch_size, num_heads, seq_len_q, d_v)
        torch.Tensor: The attention weights, shape (batch_size, num_heads, seq_len_q, seq_len_k)
    """
    d_k = query.shape[-1]  # Dimension of keys/queries

    # 1. Calculate Attention Scores: (Q @ K^T)
    # Transpose key tensor for matrix multiplication: (..., seq_len_k, d_k) -> (..., d_k, seq_len_k)
    # Result shape: (batch_size, num_heads, seq_len_q, seq_len_k)
    attention_scores = torch.matmul(query, key.transpose(-2, -1))

    # 2. Scale the scores
    attention_scores = attention_scores / math.sqrt(d_k)

    # 3. Apply Mask (if provided)
    if mask is not None:
        # Apply the mask by filling positions with a large negative value (-1e9).
        # This ensures that these positions get near-zero probability after softmax.
        # The mask should have True/1 where positions should be masked.
        attention_scores = attention_scores.masked_fill(mask == True, -1e9) # Or mask == 1

    # 4. Apply Softmax to get Attention Weights
    # Softmax is applied on the last dimension (seq_len_k) to get probabilities over keys.
    # Shape: (batch_size, num_heads, seq_len_q, seq_len_k)
    attention_weights = torch.softmax(attention_scores, dim=-1)

    # 5. Apply Dropout (optional)
    if dropout is not None:
        attention_weights = dropout(attention_weights)

    # 6. Multiply weights by Values: (Weights @ V)
    # Shape: (batch_size, num_heads, seq_len_q, d_v)
    output = torch.matmul(attention_weights, value)

    return output, attention_weights