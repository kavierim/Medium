# --- 1. Setup & Imports ---
# Make sure necessary libraries are installed
# !pip install transformers datasets evaluate torch -q
# !pip install 'accelerate>=0.26.0' -q
# 'accelerate' helps manage training on different hardware (CPU/GPU/TPU)

import datasets
import transformers
import evaluate
import torch
import numpy as np
from transformers import pipeline 

print(f"Using Datasets version: {datasets.__version__}")
print(f"Using Transformers version: {transformers.__version__}")
print(f"Using Evaluate version: {evaluate.__version__}")
print(f"Using PyTorch version: {torch.__version__}")

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- 2. Load Dataset ---
# We'll use a small subset of the IMDB dataset for faster demonstration
# You can replace 'imdb' with other datasets like 'sst2' from GLUE, or load your own CSV/JSON
print("\nLoading dataset...")
# Load only 1000 examples for training and 1000 for testing to speed things up
train_ds = datasets.load_dataset("imdb", split="train[:1000]")
eval_ds = datasets.load_dataset("imdb", split="test[:1000]")

# Inspect the dataset
print("\nDataset structure:")
print(train_ds)
print("\nExample entry:")
print(train_ds[0])

# --- 3. Load Pre-trained Model & Tokenizer ---
# Choose a base model. DistilBERT is smaller and faster than BERT.
model_name = "distilbert-base-uncased"
print(f"\nLoading tokenizer and model for: {model_name}")

tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

# Load the model for sequence classification.
# num_labels should match the number of unique labels in our dataset (positive/negative -> 2)
model = transformers.AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
model.to(device) # Move model to the GPU if available

# --- 4. Preprocess/Tokenize Dataset ---
print("\nTokenizing dataset...")
# Create a function to tokenize the text
def tokenize_function(examples):
    # padding='max_length' pads to the model's max input size
    # truncation=True cuts sequences longer than max length
    return tokenizer(examples["text"], padding="max_length", truncation=True)

# Apply the tokenization function to the entire dataset using .map()
# batched=True processes multiple examples at once for speed
tokenized_train_ds = train_ds.map(tokenize_function, batched=True)
tokenized_eval_ds = eval_ds.map(tokenize_function, batched=True)

# Remove the original 'text' column as it's no longer needed
tokenized_train_ds = tokenized_train_ds.remove_columns(["text"])
tokenized_eval_ds = tokenized_eval_ds.remove_columns(["text"])

# Rename the 'label' column to 'labels' (expected by the Trainer)
tokenized_train_ds = tokenized_train_ds.rename_column("label", "labels")
tokenized_eval_ds = tokenized_eval_ds.rename_column("label", "labels")

# Set the format to PyTorch tensors
tokenized_train_ds.set_format("torch")
tokenized_eval_ds.set_format("torch")

print("\nTokenized dataset example:")
print(tokenized_train_ds[0])

# --- 5. Define Evaluation Metrics ---
print("\nDefining evaluation metric (accuracy)...")
metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    # Get the predictions by finding the index with the highest logit
    predictions = np.argmax(logits, axis=-1)
    # Compute the accuracy
    return metric.compute(predictions=predictions, references=labels)

# --- 6. Define Training Arguments ---
print("\nDefining training arguments...")
training_args = transformers.TrainingArguments(
    output_dir="./results",          # Directory to save the model and logs
    num_train_epochs=1,              # Reduce epochs for faster demo (usually 3-5)
    per_device_train_batch_size=8,   # Reduce batch size if memory is limited
    per_device_eval_batch_size=8,
    warmup_steps=100,                # Number of steps for learning rate warmup
    weight_decay=0.01,               # Strength of weight decay regularization
    logging_dir='./logs',            # Directory for storing logs
    logging_steps=10,                # Log training info every 10 steps
    eval_strategy="epoch",           # Changed from evaluation_strategy to eval_strategy
    save_strategy="epoch",           # Save model checkpoint at the end of each epoch
    load_best_model_at_end=True,     # Load the best model found during training
    metric_for_best_model="accuracy",# Use accuracy to determine the best model
    # push_to_hub=False,             # Set to True to upload model to Hugging Face Hub
)

# --- 7. Instantiate the Trainer ---
print("\nInstantiating Trainer...")
trainer = transformers.Trainer(
    model=model,                         # The model to train
    args=training_args,                  # Training arguments
    train_dataset=tokenized_train_ds,    # Training dataset
    eval_dataset=tokenized_eval_ds,      # Evaluation dataset
    compute_metrics=compute_metrics,     # Function to compute metrics
    tokenizer=tokenizer,                 # Tokenizer (needed for padding/batching)
)

# --- 8. Train (Fine-tune) ---
print("\nStarting fine-tuning...")
try:
    train_result = trainer.train()
    print("\nFine-tuning finished.")
    # Log some training metrics
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state() # Saves trainer state (important for resuming)
    trainer.save_model("./results/best_model") # Explicitly save the best model

except Exception as e:
    print(f"\nAn error occurred during training: {e}")
    print("Fine-tuning might require significant resources (GPU recommended).")

# --- 9. Evaluate ---
print("\nEvaluating the fine-tuned model...")
try:
    eval_metrics = trainer.evaluate()
    print(f"\nEvaluation results:")
    print(eval_metrics)
    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)

except Exception as e:
    print(f"\nAn error occurred during evaluation: {e}")


# --- 10. Using the Fine-Tuned Model (Example) ---
print("\nUsing the fine-tuned model for inference...")
try:
    # Load the fine-tuned model using pipeline for simplicity
    # Make sure to specify the directory where the best model was saved
    fine_tuned_pipeline = pipeline("sentiment-analysis", model="./results/best_model", device=0 if torch.cuda.is_available() else -1)

    test_text_positive = "This is the best movie I have seen in years!"
    test_text_negative = "What a waste of time, the plot was terrible."

    print(f"Test Positive: '{test_text_positive}' -> Prediction: {fine_tuned_pipeline(test_text_positive)}")
    print(f"Test Negative: '{test_text_negative}' -> Prediction: {fine_tuned_pipeline(test_text_negative)}")

except Exception as e:
    print(f"\nCould not load or run inference with the fine-tuned model: {e}")
    print("Ensure the model was saved correctly in './results/best_model'.")

