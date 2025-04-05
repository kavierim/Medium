# Make sure you have run: pip install transformers torch datasets evaluate
from transformers import pipeline

# Load the sentiment analysis pipeline
# This will download a default pre-trained model for sentiment analysis if not cached
sentiment_pipeline = pipeline("sentiment-analysis")

# Example sentences
texts = [
    "This movie was absolutely fantastic! Highly recommended.",
    "I waited an hour for my food, and it was cold. Terrible experience.",
    "The weather today is quite pleasant.",
    "Transformers are revolutionizing artificial intelligence." # More neutral example
]

# Perform sentiment analysis
results = sentiment_pipeline(texts)

# Print the results clearly
print("Sentiment Analysis Results:")
for text, result in zip(texts, results):
    print(f"- Text: \"{text}\"")
    print(f"  Label: {result['label']}, Score: {result['score']:.4f}")

