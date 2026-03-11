import torch
import sys
import os

# Set up paths to import from parent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

import config
from data.dataset import vocab_size, decode
from model.gpt import GPTLanguageModel

# Ensure the model file exists before trying to load it
model_path = os.path.join(current_dir, config.checkpoint_path)
if not os.path.exists(model_path):
    print(f"Error: Model weights not found at {model_path}")
    print("Please run `python train/train_gpt.py` first to train and save the model.")
    sys.exit(1)

# Instantiate the blank model structure
print("Initializing model...")
model = GPTLanguageModel(vocab_size)

# Load the saved state dictionary
print(f"Loading weights from {config.checkpoint_path}...")
model.load_state_dict(torch.load(model_path, map_location=config.device, weights_only=True))

# Move it to the correct device (CPU, CUDA, or MPS)
m = model.to(config.device)
m.eval() # Set model to evaluation mode

# Print parameters
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

print("\n--- Generating some text ---")
# Start with a single blank token (0) as the context
context = torch.zeros((1, 1), dtype=torch.long, device=config.device)

# Generate and decode!
generated_tokens = m.generate(context, max_new_tokens=500)[0].tolist()
print(decode(generated_tokens))
