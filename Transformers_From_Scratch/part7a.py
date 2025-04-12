import torch
import torch.nn as nn

class ProjectionLayer(nn.Module):
    """
    Implements the final linear layer to project Decoder output to vocabulary size.
    """
    def __init__(self, d_model: int, vocab_size: int):
        """
        Args:
            d_model (int): Dimension of the model (input features).
            vocab_size (int): Size of the target vocabulary (output features).
        """
        super().__init__()
        # A simple linear layer
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor):
        """
        Forward pass for the projection layer.

        Args:
            x (torch.Tensor): Input tensor from the Decoder stack,
                              shape (batch_size, seq_len, d_model).

        Returns:
            torch.Tensor: Output tensor (logits), shape (batch_size, seq_len, vocab_size).
        """
        # Project the input tensor to the vocabulary size dimension
        # (batch_size, seq_len, d_model) -> (batch_size, seq_len, vocab_size)
        return self.proj(x)