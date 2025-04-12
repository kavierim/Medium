import torch
import torch.nn as nn

# Assume all previously defined modules are available/imported:
# InputEmbeddings, PositionalEncoding, MultiHeadAttention, PositionwiseFeedForward,
# ResidualConnection, EncoderBlock, Encoder, DecoderBlock, Decoder, ProjectionLayer

from part1a import InputEmbeddings
from part1b import PositionalEncoding
from part5b import Encoder
from part6b import Decoder
from part7a import ProjectionLayer

class Transformer(nn.Module):
    """
    The complete Transformer model integrating Encoder, Decoder, Embeddings, etc.
    """
    def __init__(self,
                 encoder: Encoder,
                 decoder: Decoder,
                 src_embed: InputEmbeddings,
                 tgt_embed: InputEmbeddings,
                 src_pos: PositionalEncoding,
                 tgt_pos: PositionalEncoding,
                 projection_layer: ProjectionLayer):
        """
        Args:
            encoder (Encoder): The Encoder stack.
            decoder (Decoder): The Decoder stack.
            src_embed (InputEmbeddings): Source embedding layer.
            tgt_embed (InputEmbeddings): Target embedding layer.
            src_pos (PositionalEncoding): Source positional encoding layer.
            tgt_pos (PositionalEncoding): Target positional encoding layer.
            projection_layer (ProjectionLayer): Final projection layer.
        """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor):
        """
        Performs the encoding part of the Transformer.

        Args:
            src (torch.Tensor): Source input tensor (token IDs), shape (batch_size, src_seq_len).
            src_mask (torch.Tensor): Source mask.

        Returns:
            torch.Tensor: Encoder output, shape (batch_size, src_seq_len, d_model).
        """
        # Apply source embedding and positional encoding
        src = self.src_embed(src)
        src = self.src_pos(src)
        # Pass through the encoder stack
        return self.encoder(src, src_mask)

    def decode(self, encoder_output: torch.Tensor, src_mask: torch.Tensor,
               tgt: torch.Tensor, tgt_mask: torch.Tensor):
        """
        Performs the decoding part of the Transformer.

        Args:
            encoder_output (torch.Tensor): Output from the encoder.
            src_mask (torch.Tensor): Source mask (for cross-attention).
            tgt (torch.Tensor): Target input tensor (token IDs), shape (batch_size, tgt_seq_len).
            tgt_mask (torch.Tensor): Target mask (for self-attention).

        Returns:
            torch.Tensor: Decoder output, shape (batch_size, tgt_seq_len, d_model).
        """
        # Apply target embedding and positional encoding
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        # Pass through the decoder stack, providing encoder output and masks
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)

    def project(self, x: torch.Tensor):
        """
        Applies the final projection layer.

        Args:
            x (torch.Tensor): Input tensor (usually from decoder), shape (batch_size, seq_len, d_model).

        Returns:
            torch.Tensor: Output logits, shape (batch_size, seq_len, tgt_vocab_size).
        """
        return self.projection_layer(x)

    # Optional: Define a full forward pass if needed for simpler training loops,
    # but separate encode/decode is often more flexible for inference.
    # def forward(self, src, tgt, src_mask, tgt_mask):
    #     encoder_output = self.encode(src, src_mask)
    #     decoder_output = self.decode(encoder_output, src_mask, tgt, tgt_mask)
    #     logits = self.project(decoder_output)
    #     return logits