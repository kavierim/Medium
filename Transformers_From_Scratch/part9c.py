import torch
import random

# Assume Transformer class and all sub-modules are defined/imported
# Assume greedy_decode, create_src_mask, tokenize_sequence, id_to_token, etc. are defined
# Assume VOCAB_SIZE, PAD_ID, SOS_TOKEN, EOS_TOKEN, token_to_id, id_to_token are defined
# Assume device is defined
from part9b import *

# --- Load Trained Model (Example) ---
print("\n--- Running Inference ---")

# 1. Instantiate model architecture (must match training)
d_model_inf = 128
num_layers_inf = 3
num_heads_inf = 4
d_ff_inf = 512
dropout_inf = 0.1
max_len_inf = MAX_PADDED_LEN # Use same max length

src_embed_inf = InputEmbeddings(d_model_inf, VOCAB_SIZE)
tgt_embed_inf = InputEmbeddings(d_model_inf, VOCAB_SIZE)
src_pos_inf = PositionalEncoding(d_model_inf, max_len_inf, dropout_inf)
tgt_pos_inf = PositionalEncoding(d_model_inf, max_len_inf, dropout_inf)
attention_inf = MultiHeadAttention(d_model_inf, num_heads_inf, dropout_inf)
ff_inf = PositionwiseFeedForward(d_model_inf, d_ff_inf, dropout_inf)
proj_inf = ProjectionLayer(d_model_inf, VOCAB_SIZE)

encoder_blocks_inf = nn.ModuleList([ EncoderBlock(d_model_inf, copy.deepcopy(attention_inf), copy.deepcopy(ff_inf), dropout_inf) for _ in range(num_layers_inf) ])
decoder_blocks_inf = nn.ModuleList([ DecoderBlock(d_model_inf, copy.deepcopy(attention_inf), copy.deepcopy(attention_inf), copy.deepcopy(ff_inf), dropout_inf) for _ in range(num_layers_inf) ])

encoder_inf = Encoder(d_model_inf, encoder_blocks_inf)
decoder_inf = Decoder(d_model_inf, decoder_blocks_inf)

inference_model = Transformer(
    encoder=encoder_inf, decoder=decoder_inf,
    src_embed=src_embed_inf, tgt_embed=tgt_embed_inf,
    src_pos=src_pos_inf, tgt_pos=tgt_pos_inf,
    projection_layer=proj_inf
).to(device)

# 2. Load saved weights (if you saved them in Part 8)
# Example:
try:
    model_state_path = "transformer_copy_task.pth"
    inference_model.load_state_dict(torch.load(model_state_path, map_location=device))
    print(f"Loaded trained model weights from {model_state_path}")
except FileNotFoundError:
    print("WARNING: Trained model weights not found. Using randomly initialized model for inference.")
except Exception as e:
    print(f"WARNING: Error loading model weights: {e}. Using randomly initialized model.")

# Set to eval mode AFTER loading weights
inference_model.eval()

# 3. Prepare Input Sequence
# Take a random example or define one
# src_text = ['[SOS]', '3', '1', '4', '1', '5', '[EOS]']
src_text = random.choice(raw_data)['src'] # Use one from generated data

print(f"\nInput Sequence (Text): {src_text}")

# Tokenize and convert to tensor (add batch dimension)
src_ids_inf = torch.tensor([pad_sequence(tokenize_sequence(src_text, token_to_id), max_len_inf, PAD_ID)], dtype=torch.long)
print(f"Input Sequence (IDs): {src_ids_inf}")

# 4. Create Source Mask
src_mask_inf = create_src_mask(src_ids_inf, PAD_ID)
print(f"Source Mask shape: {src_mask_inf.shape}")

# 5. Run Greedy Decoding
sos_id = token_to_id[SOS_TOKEN]
eos_id = token_to_id[EOS_TOKEN]

print("\nGenerating output using Greedy Decoding...")
generated_ids = greedy_decode(inference_model, src_ids_inf, src_mask_inf, max_len=max_len_inf, sos_id=sos_id, eos_id=eos_id, device=device)
print(f"Generated Sequence (IDs): {generated_ids}")

# 6. Convert Output IDs to Tokens
generated_tokens = [id_to_token[id_] for id_ in generated_ids[0].cpu().numpy()] # Get item from batch, move to CPU
print(f"Generated Sequence (Text): {generated_tokens}")

# For copy task, ideally generated_tokens should match src_text
if generated_tokens == src_text:
    print("\nSuccess! Generated sequence matches input sequence.")
else:
    print("\nGenerated sequence does NOT perfectly match input sequence (may need more training or better decoding).")

