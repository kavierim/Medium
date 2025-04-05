# Make sure you have run: pip install transformers torch sentencepiece
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load a model suitable for causal language modeling (text generation)
model_name = "gpt2" # You can try "distilgpt2" for a smaller version
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Input prompt
prompt = "The future of artificial intelligence"

# Tokenize the prompt
inputs = tokenizer(prompt, return_tensors="pt")
input_ids = inputs['input_ids']

# --- Decoding Strategies ---

print(f"Prompt: \"{prompt}\"")

# 1. Greedy Search (Default - picks the most likely word at each step)
print("\n--- Greedy Search ---")
with torch.no_grad():
    greedy_output = model.generate(input_ids, max_length=50)
print(f"Output:\n{tokenizer.decode(greedy_output[0], skip_special_tokens=True)}")
# Note: Greedy search can sometimes lead to repetitive loops.

# 2. Beam Search (Explores multiple possibilities or 'beams')
print("\n--- Beam Search (num_beams=5) ---")
with torch.no_grad():
    beam_output = model.generate(
        input_ids,
        max_length=50,
        num_beams=5, # Keep track of the 5 most likely sequences
        early_stopping=True # Stop beams early if end-of-sequence is reached
    )
print(f"Output:\n{tokenizer.decode(beam_output[0], skip_special_tokens=True)}")
# Note: Beam search often produces more fluent text but can be less surprising.

# 3. Sampling (Introduces randomness)
print("\n--- Sampling (do_sample=True) ---")
# Set pad_token_id to eos_token_id for open-end generation with sampling
model.config.pad_token_id = model.config.eos_token_id

with torch.no_grad():
    # Basic sampling
    sample_output = model.generate(
        input_ids,
        max_length=50,
        do_sample=True, # Enable sampling
        top_k=0 # Set top_k=0 to disable Top-K filtering for basic sampling
    )
print(f"Basic Sampling Output:\n{tokenizer.decode(sample_output[0], skip_special_tokens=True)}")

# 4. Sampling with Temperature (Controls randomness)
print("\n--- Sampling with Temperature (temp=0.7) ---")
with torch.no_grad():
    # Lower temperature -> less random, more focused
    sample_temp_output = model.generate(
        input_ids,
        max_length=50,
        do_sample=True,
        temperature=0.7 # Lower temp makes distribution sharper (less random)
    )
print(f"Output:\n{tokenizer.decode(sample_temp_output[0], skip_special_tokens=True)}")

# 5. Top-K Sampling (Samples from K most likely words)
print("\n--- Top-K Sampling (top_k=50) ---")
with torch.no_grad():
    sample_topk_output = model.generate(
        input_ids,
        max_length=50,
        do_sample=True,
        top_k=50 # Consider only the 50 most likely words at each step
    )
print(f"Output:\n{tokenizer.decode(sample_topk_output[0], skip_special_tokens=True)}")

# 6. Top-P (Nucleus) Sampling (Samples from smallest set exceeding probability P)
print("\n--- Top-P Sampling (top_p=0.9) ---")
with torch.no_grad():
    sample_topp_output = model.generate(
        input_ids,
        max_length=50,
        do_sample=True,
        top_p=0.9, # Sample from the smallest set of words whose cumulative probability >= 0.9
        top_k=0 # Important: Set top_k=0 when using top_p
    )
print(f"Output:\n{tokenizer.decode(sample_topp_output[0], skip_special_tokens=True)}")
