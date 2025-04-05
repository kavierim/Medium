# Make sure you have run: pip install transformers torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch

# Choose a pre-trained QA model
# distilbert-base-cased-distilled-squad is a smaller, faster QA model
model_name = "distilbert-base-cased-distilled-squad"

# Load the tokenizer and model specifically designed for Question Answering
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

# Same context and question
context = """
The Apollo 11 mission, launched on July 16, 1969, was the first mission to land humans on the Moon.
The commander was Neil Armstrong, the command module pilot was Michael Collins, and the lunar module pilot was Buzz Aldrin.
Armstrong became the first person to step onto the lunar surface on July 21, 1969, followed by Aldrin.
"""
question = "Who was the commander of the Apollo 11 mission?"

# 1. Tokenize the input (question and context together)
# The tokenizer handles formatting them correctly for the model
inputs = tokenizer(question, context, return_tensors="pt")

# 2. Perform inference
with torch.no_grad():
    outputs = model(**inputs)
    # The outputs contain 'start_logits' and 'end_logits'

# 3. Get the most likely start and end token positions
start_logits = outputs.start_logits
end_logits = outputs.end_logits

# Find the token indices with the highest start and end scores
start_index = torch.argmax(start_logits)
end_index = torch.argmax(end_logits)

# Ensure start_index comes before end_index
if start_index > end_index:
    print("Warning: Predicted start index is after end index. Check model/input.")
    # Basic fallback: maybe swap them or consider the highest overall logit?
    # For simplicity here, we'll proceed but note the issue.

# 4. Decode the answer span from the token indices
# We need the input_ids to map indices back to tokens
input_ids = inputs["input_ids"][0]
answer_tokens = input_ids[start_index : end_index + 1] # Slice the token IDs for the answer

# Use the tokenizer to convert token IDs back to a string
answer = tokenizer.decode(answer_tokens, skip_special_tokens=True)

print("\nManual Processing Results:")
print(f"Question: {question}")
# print(f"Predicted Start Token Index: {start_index.item()}") # .item() gets Python number from tensor
# print(f"Predicted End Token Index: {end_index.item()}")
print(f"Decoded Answer: {answer}")

# Note: More robust decoding might handle cases where the answer is impossible
# (e.g., start/end logits are very low, start > end, answer spans across context/question boundary)
print("\nNote: Manual decoding requires careful handling of token indices and potential edge cases.")

