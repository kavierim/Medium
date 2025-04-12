from part8a import *
from part8b import *

def tokenize_sequence(sequence: list, token_to_id_map: dict) -> list:
    """Converts a sequence of tokens to a sequence of IDs."""
    return [token_to_id_map[token] for token in sequence]

def pad_sequence(sequence_ids: list, max_len: int, pad_id: int) -> list:
    """Pads a sequence of IDs to max_len."""
    padded_ids = sequence_ids + [pad_id] * (max_len - len(sequence_ids))
    return padded_ids[:max_len] # Ensure it doesn't exceed max_len

# Define max length for padding (should accommodate SOS, EOS, and max content length)
MAX_PADDED_LEN = MAX_SEQ_LEN + 2 # +2 for SOS and EOS

# Example tokenization and padding
example = raw_data[0]
src_ids = tokenize_sequence(example['src'], token_to_id)
tgt_ids = tokenize_sequence(example['tgt'], token_to_id)
src_padded = pad_sequence(src_ids, MAX_PADDED_LEN, PAD_ID)
tgt_padded = pad_sequence(tgt_ids, MAX_PADDED_LEN, PAD_ID)

print(f"\nOriginal Src: {example['src']}")
print(f"Tokenized Src: {src_ids}")
print(f"Padded Src: {src_padded}")
print(f"Original Tgt: {example['tgt']}")
print(f"Tokenized Tgt: {tgt_ids}")
print(f"Padded Tgt: {tgt_padded}")