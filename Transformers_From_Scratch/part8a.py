# Define special tokens and vocabulary
PAD_TOKEN = '[PAD]'
SOS_TOKEN = '[SOS]' # Start Of Sequence
EOS_TOKEN = '[EOS]' # End Of Sequence

# Vocabulary for digits 0-9 plus special tokens
VOCAB = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN] + [str(i) for i in range(10)]

# Create token-to-ID and ID-to-token mappings
token_to_id = {token: i for i, token in enumerate(VOCAB)}
id_to_token = {i: token for token, i in token_to_id.items()}

VOCAB_SIZE = len(VOCAB)
PAD_ID = token_to_id[PAD_TOKEN]

print(f"Vocabulary: {VOCAB}")
print(f"Vocab Size: {VOCAB_SIZE}")
print(f"Token to ID: {token_to_id}")
print(f"PAD ID: {PAD_ID}")