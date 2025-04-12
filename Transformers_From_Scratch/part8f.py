import torch
import torch.nn as nn

from part8a import *

# Loss Function (ignore padding)
loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_ID)

# Optimizer (Adam) - requires model parameters
# We will instantiate the model first, then the optimizer
# Example: optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.98), eps=1e-9)
LEARNING_RATE = 1e-4 # Example learning rate
