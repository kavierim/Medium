import torch
import torch.nn as nn
import math

class InputEmbeddings(nn.Module):
    """
    Converts input token IDs to dense vector embeddings.
    Also multiplies the embeddings by sqrt(d_model) as per the paper.
    """
    def __init__(self, d_model: int, vocab_size: int):
        """
        Args:
            d_model (int): Dimension of the embeddings (and the model).
            vocab_size (int): Size of the vocabulary.
        """
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        # Create the embedding layer
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        """
        Forward pass of the embedding layer.

        Args:
            x (torch.Tensor): Input tensor of token IDs (batch_size, seq_len).

        Returns:
            torch.Tensor: Output tensor of embeddings (batch_size, seq_len, d_model).
        """
        # Look up embeddings and scale them
        # Paper Section 3.4: "In the embedding layers, we multiply those weights by sqrt(d_model)."
        return self.embedding(x) * math.sqrt(self.d_model)
    