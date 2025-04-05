# Make sure you have run: pip install transformers torch sentencepiece
from transformers import pipeline

# Load the text generation pipeline (using gpt2 by default if not specified)
# You can specify other models like 'gpt2-medium', 'distilgpt2', etc.
generator = pipeline('text-generation', model='gpt2')

# Starting prompt
prompt = "In a world where robots cook breakfast,"

# Generate text
# max_length controls the total length of the output (prompt + generated text)
# num_return_sequences generates multiple different continuations
generated_texts = generator(prompt, max_length=50, num_return_sequences=2)

# Print the results
print(f"Prompt: \"{prompt}\"")
print("\nGenerated Texts:")
if generated_texts:
    for i, text in enumerate(generated_texts):
        print(f"{i+1}: {text['generated_text']}")
else:
    print("Text generation failed.")

