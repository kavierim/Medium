# Make sure you have run: pip install transformers torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch # Import torch if you haven't already

# Choose a pre-trained NER model
# dslim/bert-base-NER is a popular fine-tuned BERT model for NER
model_name = "dslim/bert-base-NER"

# Load the tokenizer and model specifically designed for Token Classification (like NER)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)

# Same example text
text = "My name is Clara and I live in Berlin. I work for the European Space Agency."

# 1. Tokenize the input text
# return_tensors='pt' gives us PyTorch tensors
inputs = tokenizer(text, return_tensors="pt")

# 2. Perform inference (get model predictions)
# We use torch.no_grad() to disable gradient calculations for inference (saves memory)
with torch.no_grad():
    outputs = model(**inputs)

# 3. Process the outputs
# Outputs contain 'logits' - raw scores for each possible tag for each token
predictions = torch.argmax(outputs.logits, dim=2)

# 4. Map predictions back to labels and words (Simplified Example)
print("\nManual Processing Results (Token Level):")
tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0]) # Get token strings
predicted_labels = [model.config.id2label[prediction.item()] for prediction in predictions[0]] # Map prediction IDs to label names

# Print tokens and their predicted labels
# Note: This doesn't automatically group entities like the pipeline did.
# It shows the raw prediction for each token (including subwords like ##ara for Clara).
for token, label in zip(tokens, predicted_labels):
    # Ignore special tokens like [CLS], [SEP] for clarity
    if token not in ['[CLS]', '[SEP]']:
        print(f"- Token: {token}, Predicted Label: {label}")

# To get grouped entities like the pipeline, more complex logic is needed
# to handle the BIO tagging scheme (e.g., B-PER, I-PER) and combine subwords.
# This is often handled by helper functions or the pipeline itself.
print("\nNote: Grouping entities from raw token predictions requires additional logic.")
