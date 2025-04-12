import time
import copy
import torch
import torch.nn as nn
# Assume Transformer class and all sub-modules are defined/imported
# Assume create_src_mask, create_tgt_mask are defined
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
from part8d import *
from part8e import *
from part8f import *

# --- Main Training Script ---
device = "cpu" # Change to "cuda" if GPU is available 

def train_one_epoch(model, dataloader, loss_fn, optimizer, device, pad_id):
    """ Trains the model for one epoch. """
    model.train() # Set model to training mode
    total_loss = 0
    start_time = time.time()

    for i, batch in enumerate(dataloader):
        # Move data to device
        src_ids = batch['src_ids'].to(device)
        decoder_input_ids = batch['decoder_input_ids'].to(device)
        label_ids = batch['label_ids'].to(device)

        # Create masks
        src_mask = create_src_mask(src_ids, pad_id).to(device)
        tgt_mask = create_tgt_mask(decoder_input_ids, pad_id).to(device) # Use decoder_input for tgt mask shape

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        encoder_output = model.encode(src_ids, src_mask)
        decoder_output = model.decode(encoder_output, src_mask, decoder_input_ids, tgt_mask)
        logits = model.project(decoder_output) # Shape: [Batch, SeqLen, VocabSize]

        # Calculate loss
        # Reshape for CrossEntropyLoss: [Batch * SeqLen, VocabSize] and [Batch * SeqLen]
        loss = loss_fn(logits.view(-1, logits.shape[-1]), label_ids.view(-1))

        # Backward pass
        loss.backward()

        # Optimizer step
        optimizer.step()

        total_loss += loss.item()

        # Optional: Print progress
        if (i + 1) % 10 == 0: # Print every 10 batches
             elapsed = time.time() - start_time
             print(f"  Batch {i+1}/{len(dataloader)}, Loss: {loss.item():.4f}, Time: {elapsed:.2f}s")
             start_time = time.time() # Reset timer

    avg_loss = total_loss / len(dataloader)
    return avg_loss



# 1. Instantiate the Full Model (from Part 7)
#    Make sure all parameters match (d_model, heads, layers, vocab sizes etc.)
print("\n--- Setting up Model for Training ---")
d_model = 128 # Smaller model for faster demo
num_layers = 3 # Fewer layers
num_heads = 4
d_ff = 512
dropout = 0.1
max_len = MAX_PADDED_LEN # Use padded length for PE

src_embed_train = InputEmbeddings(d_model, VOCAB_SIZE)
tgt_embed_train = InputEmbeddings(d_model, VOCAB_SIZE) # Same vocab for copy task
src_pos_train = PositionalEncoding(d_model, max_len, dropout)
tgt_pos_train = PositionalEncoding(d_model, max_len, dropout)
attention_train = MultiHeadAttention(d_model, num_heads, dropout)
ff_train = PositionwiseFeedForward(d_model, d_ff, dropout)
proj_train = ProjectionLayer(d_model, VOCAB_SIZE)

encoder_blocks_train = nn.ModuleList([
    EncoderBlock(d_model, copy.deepcopy(attention_train), copy.deepcopy(ff_train), dropout) for _ in range(num_layers)
])
decoder_blocks_train = nn.ModuleList([
    DecoderBlock(d_model, copy.deepcopy(attention_train), copy.deepcopy(attention_train), copy.deepcopy(ff_train), dropout) for _ in range(num_layers)
])

encoder_train = Encoder(d_model, encoder_blocks_train)
decoder_train = Decoder(d_model, decoder_blocks_train)

model = Transformer(
    encoder=encoder_train, decoder=decoder_train,
    src_embed=src_embed_train, tgt_embed=tgt_embed_train,
    src_pos=src_pos_train, tgt_pos=tgt_pos_train,
    projection_layer=proj_train
).to(device) # Move model to device

# Initialize weights
for p in model.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)

print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters.")

# 2. Define Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.98), eps=1e-9)

# 3. Run Training Loop
NUM_EPOCHS = 5 # Train for a few epochs for demo
print(f"\n--- Starting Training for {NUM_EPOCHS} Epochs ---")

for epoch in range(1, NUM_EPOCHS + 1):
    epoch_start_time = time.time()
    avg_epoch_loss = train_one_epoch(model, train_dataloader, loss_fn, optimizer, device, PAD_ID)
    epoch_duration = time.time() - epoch_start_time
    print("-" * 50)
    print(f"End of Epoch {epoch} | Time: {epoch_duration:.2f}s | Avg Loss: {avg_epoch_loss:.4f}")
    print("-" * 50)

print("Training finished.")
# Optional: Save the trained model
# torch.save(model.state_dict(), "transformer_copy_task.pth")