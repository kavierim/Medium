# Make sure you have run: pip install transformers torch sentencepiece
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# Choose a specific translation model (e.g., English to French)
# Helsinki-NLP models are excellent for translation
model_name = "Helsinki-NLP/opus-mt-en-fr"

# Load the tokenizer and the Seq2Seq model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Input text
text_en = "Machine learning is a fascinating field."

# 1. Tokenize the input text
# Prepare the text for the model (convert to token IDs)
inputs = tokenizer(text_en, return_tensors="pt")

# 2. Generate the translation using model.generate()
# This performs the core Seq2Seq generation process
with torch.no_grad():
    output_sequences = model.generate(
        input_ids=inputs['input_ids'],
        attention_mask=inputs['attention_mask'], # Include attention mask
        # You can add parameters like max_length, num_beams, etc. here
        # max_length=50,
        # num_beams=4,
        # early_stopping=True
    )

# 3. Decode the generated token IDs back to text
# Use the tokenizer to convert the output IDs to a readable string
translation_fr = tokenizer.decode(output_sequences[0], skip_special_tokens=True)

print("\nManual Processing Results:")
print(f"Original (en): \"{text_en}\"")
print(f"Translation (fr): \"{translation_fr}\"")

