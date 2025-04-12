

import torch
import torch.nn as nn
import copy # For creating deep copies of layers

# Assume all module classes are defined/imported from previous parts:
# InputEmbeddings, PositionalEncoding, MultiHeadAttention, PositionwiseFeedForward,
# ResidualConnection, LayerNormalization, EncoderBlock, Encoder, DecoderBlock, Decoder, ProjectionLayer
from part1a import InputEmbeddings
from part1b import PositionalEncoding
from part3a import MultiHeadAttention
from part4b import PositionwiseFeedForward
from part5a import EncoderBlock
from part5b import Encoder
from part6a import DecoderBlock
from part6b import Decoder
from part7a import ProjectionLayer
from part7b import Transformer


print("\n--- Testing Full Transformer Model ---")

# Model Parameters (Matching Base Paper)
src_vocab_size = 10000 # Example source vocab size
tgt_vocab_size = 11000 # Example target vocab size
d_model = 512
num_layers = 6 # N=6
num_heads = 8
dropout = 0.1
d_ff = 2048
max_seq_len = 100 # Max sequence length for positional encoding

# --- Create instances of the building blocks ---

# Embeddings & Positional Encoding
src_embed = InputEmbeddings(d_model, src_vocab_size)
tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)
src_pos = PositionalEncoding(d_model, max_seq_len, dropout)
tgt_pos = PositionalEncoding(d_model, max_seq_len, dropout)

# Attention, FFN, Residuals (can be shared or separate per block/layer)
attention = MultiHeadAttention(d_model, num_heads, dropout)
feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
# Note: ResidualConnection takes features and dropout, sublayer is passed in forward

# Encoder Blocks & Stack
encoder_blocks = nn.ModuleList([
    EncoderBlock(d_model, copy.deepcopy(attention), copy.deepcopy(feed_forward), dropout)
    for _ in range(num_layers)
])
encoder = Encoder(d_model, encoder_blocks)

# Decoder Blocks & Stack
decoder_blocks = nn.ModuleList([
    DecoderBlock(d_model, copy.deepcopy(attention), copy.deepcopy(attention), copy.deepcopy(feed_forward), dropout)
    for _ in range(num_layers)
])
decoder = Decoder(d_model, decoder_blocks)

# Projection Layer
projection = ProjectionLayer(d_model, tgt_vocab_size)

# --- Instantiate the Full Transformer ---
transformer_model = Transformer(
    encoder=encoder,
    decoder=decoder,
    src_embed=src_embed,
    tgt_embed=tgt_embed,
    src_pos=src_pos,
    tgt_pos=tgt_pos,
    projection_layer=projection
)

# Initialize parameters (Xavier uniform is common)
for p in transformer_model.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)

print(f"Transformer model created with {sum(p.numel() for p in transformer_model.parameters()):,} parameters.")

# --- Test with Dummy Data ---
batch_size_test = 2
src_seq_len_test = 10
tgt_seq_len_test = 12

# Dummy source and target sequences (token IDs)
dummy_src_seq = torch.randint(1, src_vocab_size, (batch_size_test, src_seq_len_test)) # Avoid 0 if it's padding
dummy_tgt_seq = torch.randint(1, tgt_vocab_size, (batch_size_test, tgt_seq_len_test))

# Dummy masks (False where token is present, True where padded/masked)
# Simple example: no padding, just look-ahead for target
dummy_src_mask_test = torch.zeros(batch_size_test, 1, 1, src_seq_len_test, dtype=torch.bool)
dummy_tgt_mask_test = torch.triu(torch.ones(tgt_seq_len_test, tgt_seq_len_test), diagonal=1).bool().unsqueeze(0).unsqueeze(0) # [1, 1, T, T] broadcastable

print(f"\nDummy Source Seq shape: {dummy_src_seq.shape}")
print(f"Dummy Target Seq shape: {dummy_tgt_seq.shape}")
print(f"Dummy Source Mask shape: {dummy_src_mask_test.shape}")
print(f"Dummy Target Mask shape: {dummy_tgt_mask_test.shape}")

# --- Test encode, decode, project methods ---
transformer_model.eval() # Set model to evaluation mode
with torch.no_grad():
    print("\nTesting encode...")
    encoder_output_test = transformer_model.encode(dummy_src_seq, dummy_src_mask_test)
    print(f"Encoder Output shape: {encoder_output_test.shape}") # Expected: [B, S_src, Dm] = [2, 10, 512]
    assert encoder_output_test.shape == (batch_size_test, src_seq_len_test, d_model)

    print("\nTesting decode...")
    decoder_output_test = transformer_model.decode(encoder_output_test, dummy_src_mask_test, dummy_tgt_seq, dummy_tgt_mask_test)
    print(f"Decoder Output shape: {decoder_output_test.shape}") # Expected: [B, S_tgt, Dm] = [2, 12, 512]
    assert decoder_output_test.shape == (batch_size_test, tgt_seq_len_test, d_model)

    print("\nTesting project...")
    final_logits_test = transformer_model.project(decoder_output_test)
    print(f"Final Logits shape: {final_logits_test.shape}") # Expected: [B, S_tgt, V_tgt] = [2, 12, 11000]
    assert final_logits_test.shape == (batch_size_test, tgt_seq_len_test, tgt_vocab_size)

print("\nFull Transformer test passed (shape checks).")