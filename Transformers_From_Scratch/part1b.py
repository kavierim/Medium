import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """
    Injects positional information into the input embeddings.
    Uses sine and cosine functions of different frequencies.
    """
    def __init__(self, d_model: int, seq_len: int, dropout: float):
        """
        Args:
            d_model (int): Dimension of the embeddings (and the model).
            seq_len (int): Maximum sequence length.
            dropout (float): Dropout probability.
        """
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        # Create a positional encoding matrix of shape (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)

        # Create a tensor representing positions (0, 1, 2, ..., seq_len-1)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1) # Shape: (seq_len, 1)

        # Calculate the division term for the frequencies
        # Original formula: 10000^(2i / d_model)
        # Log space equivalent: exp((2i / d_model) * log(10000)) = exp(2i * (-log(10000.0) / d_model))
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)) # Shape: (d_model / 2)

        # Apply sine to even indices (0, 2, 4, ...)
        pe[:, 0::2] = torch.sin(position * div_term)
        # Apply cosine to odd indices (1, 3, 5, ...)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Add a batch dimension to the positional encoding matrix
        # Shape becomes (1, seq_len, d_model) so it can be easily added to input batches
        pe = pe.unsqueeze(0)

        # Register 'pe' as a buffer. Buffers are part of the model's state,
        # but are not considered parameters (not updated by the optimizer).
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Forward pass to add positional encoding.

        Args:
            x (torch.Tensor): Input tensor (batch_size, seq_len, d_model).

        Returns:
            torch.Tensor: Output tensor with positional encoding added (batch_size, seq_len, d_model).
        """
        # Add positional encoding to the input tensor x.
        # self.pe is (1, seq_len, d_model). x is (batch_size, seq_len, d_model).
        # We only need pos encoding up to the actual sequence length in the batch `x.shape[1]`.
        # The `requires_grad_(False)` ensures this slicing operation doesn't track gradients for PE.
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        # Apply dropout
        return self.dropout(x)
    