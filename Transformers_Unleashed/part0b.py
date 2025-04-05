# Make sure you've run: pip install transformers torch
from transformers import AutoTokenizer, AutoModel

try:
    # Load tokenizer associated with 'bert-base-uncased'
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    print("Tokenizer loaded successfully!")
    # Load the pre-trained model 'bert-base-uncased'
    model = AutoModel.from_pretrained('bert-base-uncased')
    print("Model loaded successfully!")
    # We won't *use* them yet, just show they can be loaded.
    print(f"Tokenizer type: {type(tokenizer)}")
    print(f"Model type: {type(model)}")
except Exception as e:
    print(f"An error occurred while loading from Hugging Face: {e}")
    print("Check your internet connection and library installation.")

