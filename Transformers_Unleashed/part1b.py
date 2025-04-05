# Make sure you have run: pip install transformers torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Specify a model known for sentiment analysis
# distilbert-base-uncased-finetuned-sst-2-english is a common choice
model_name = "distilbert-base-uncased-finetuned-sst-2-english"

# Load the tokenizer and model specifically for Sequence Classification
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Example text
text = "Hugging Face makes NLP easy!"

# 1. Tokenize the input text
inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
# return_tensors="pt" gives PyTorch tensors
# truncation=True ensures text longer than the model's max length is cut
# padding=True adds padding if the text is shorter than the max length

# 2. Perform inference (get model predictions)
with torch.no_grad(): # Disable gradient calculations for inference
    outputs = model(**inputs)

# 3. Process the outputs (logits)
logits = outputs.logits # Raw, unnormalized scores for each class
probabilities = torch.softmax(logits, dim=1) # Convert logits to probabilities
predicted_class_id = torch.argmax(probabilities, dim=1).item() # Get the class ID with highest probability

# 4. Map prediction ID to label name
predicted_label = model.config.id2label[predicted_class_id]
predicted_score = probabilities[0][predicted_class_id].item() # Get the score for the predicted class

print("\nManual Processing Results:")
print(f"- Text: \"{text}\"")
print(f"  Logits: {logits.numpy()}") # Raw scores
print(f"  Probabilities: {probabilities.numpy()}") # Probabilities for [NEG, POS] (or model specific order)
print(f"  Predicted Class ID: {predicted_class_id}")
print(f"  Predicted Label: {predicted_label}")
print(f"  Confidence Score: {predicted_score:.4f}")
