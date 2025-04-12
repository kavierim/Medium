import random

from part8a import *

def generate_copy_task_data(num_examples: int, min_len: int, max_len: int, vocab: list):
    """Generates synthetic data for the sequence copying task."""
    data = []
    # Exclude special tokens from the sequence content
    content_vocab = [token for token in vocab if token not in [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN]]
    for _ in range(num_examples):
        seq_len = random.randint(min_len, max_len)
        sequence = [random.choice(content_vocab) for _ in range(seq_len)]
        # Source and Target are the same, wrapped with SOS/EOS
        src = [SOS_TOKEN] + sequence + [EOS_TOKEN]
        tgt = [SOS_TOKEN] + sequence + [EOS_TOKEN]
        data.append({'src': src, 'tgt': tgt})
    return data

# Example generation
NUM_EXAMPLES = 1000
MIN_SEQ_LEN = 5
MAX_SEQ_LEN = 10 # Keep sequences relatively short for faster training demo
raw_data = generate_copy_task_data(NUM_EXAMPLES, MIN_SEQ_LEN, MAX_SEQ_LEN, VOCAB)
print(f"\nGenerated {len(raw_data)} examples. First example:")
print(raw_data[0])