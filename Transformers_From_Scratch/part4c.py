import torch
import torch.nn as nn

# Assuming LayerNormalization class is defined as above
from part4a import LayerNormalization

class ResidualConnection(nn.Module):
    """
    Implements the Add & Norm step: LayerNorm(x + Dropout(Sublayer(x)))
    """
    def __init__(self, features: int, dropout: float):
        """
        Args:
            features (int): Number of features (d_model).
            dropout (float): Dropout probability to apply to the sublayer output.
        """
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization(features)

    def forward(self, x: torch.Tensor, sublayer: nn.Module):
        """
        Forward pass for the residual connection.

        Args:
            x (torch.Tensor): Input tensor (batch_size, seq_len, features).
            sublayer (nn.Module): The sublayer module (e.g., MultiHeadAttention, PositionwiseFeedForward)
                                  to apply before the residual connection.

        Returns:
            torch.Tensor: Output tensor after dropout, addition, and normalization.
        """
        # Apply the sublayer, then dropout, then add the original input x (residual connection),
        # and finally apply layer normalization.
        # Note: The paper applies norm *after* the addition.
        sublayer_output = sublayer(x)
        return self.norm(x + self.dropout(sublayer_output))