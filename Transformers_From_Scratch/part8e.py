from torch.utils.data import Dataset, DataLoader

from part8a import *
from part8b import *
from part8c import *
from part8d import *

class CopyTaskDataset(Dataset):
    """ Custom Dataset for the sequence copying task. """
    def __init__(self, data, token_to_id_map, max_len, pad_id):
        self.data = data
        self.token_to_id = token_to_id_map
        self.max_len = max_len
        self.pad_id = pad_id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Tokenize and pad source sequence
        src_ids = tokenize_sequence(item['src'], self.token_to_id)
        src_padded = pad_sequence(src_ids, self.max_len, self.pad_id)

        # Tokenize target sequence
        tgt_ids = tokenize_sequence(item['tgt'], self.token_to_id)

        # Create decoder input (shifted right: SOS + sequence)
        decoder_input_ids = tgt_ids[:-1] # Exclude EOS
        decoder_input_padded = pad_sequence(decoder_input_ids, self.max_len, self.pad_id)

        # Create label (expected output: sequence + EOS)
        label_ids = tgt_ids[1:] # Exclude SOS
        label_padded = pad_sequence(label_ids, self.max_len, self.pad_id) # Pad labels too

        return {
            "src_ids": torch.tensor(src_padded, dtype=torch.long),
            "decoder_input_ids": torch.tensor(decoder_input_padded, dtype=torch.long),
            "label_ids": torch.tensor(label_padded, dtype=torch.long)
        }

# Create Dataset instance
train_dataset = CopyTaskDataset(raw_data, token_to_id, MAX_PADDED_LEN, PAD_ID)

# Create DataLoader instance
BATCH_SIZE = 64 # Adjust based on memory
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Check a batch
print("\nChecking DataLoader batch...")
batch = next(iter(train_dataloader))
print("Batch keys:", batch.keys())
print("src_ids shape:", batch['src_ids'].shape) # Expected: [BATCH_SIZE, MAX_PADDED_LEN]
print("decoder_input_ids shape:", batch['decoder_input_ids'].shape)
print("label_ids shape:", batch['label_ids'].shape)