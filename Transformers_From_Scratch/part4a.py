import torch
import torch.nn as nn
import math

class LayerNormalization(nn.Module):
    """
    Implements Layer Normalization.
    """
    def __init__(self, features: int, eps: float = 1e-6):
        """
        Args:
            features (int): Number of features (d_model).
            eps (float, optional): Epsilon value for numerical stability. Defaults to 1e-6.
        """
        super().__init__()
        self.eps = eps
        # Learnable scale parameter (gamma), initialized to ones
        self.gamma = nn.Parameter(torch.ones(features))
        # Learnable shift parameter (beta), initialized to zeros
        self.beta = nn.Parameter(torch.zeros(features))

    def forward(self, x: torch.Tensor):
        """
        Forward pass for Layer Normalization.

        Args:
            x (torch.Tensor): Input tensor, shape (batch_size, seq_len, features)

        Returns:
            torch.Tensor: Normalized tensor, shape (batch_size, seq_len, features)
        """
        # Calculate mean and standard deviation along the feature dimension (-1)
        # keepdim=True maintains the dimension for broadcasting
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)

        # Normalize the input
        # (x - mean) / sqrt(std^2 + eps)
        normalized_x = (x - mean) / torch.sqrt(std**2 + self.eps) # Using std directly: (x - mean) / (std + self.eps) is also common

        # Apply scale (gamma) and shift (beta)
        # gamma and beta have shape (features), they will broadcast correctly
        output = self.gamma * normalized_x + self.beta
        return output