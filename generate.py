import torch
import sys
import os

# Set up paths to import from parent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

import config
from shared.step1_tokenizer import encoder
from shared.step3_output import OutputHead

if config.model_version == 'gpt2':
    from model_gpt2.step2_gpt import GPTLanguageModel
elif config.model_version == 'llama':
    from model_llama.step2_gpt import GPTLanguageModel
else:
    raise ValueError(f"Unknown model_version: {config.model_version}")

# Ensure the model file exists before trying to load it
model_path = os.path.join(current_dir, config.checkpoint_path)
if not os.path.exists(model_path):
    print(f"Error: Model weights not found at {model_path}")
    print("Please run `python train/train_gpt.py` first to train and save the model.")
    sys.exit(1)

# Instantiate the blank model structure
print("Initializing model...")
model = GPTLanguageModel(encoder.vocab_size)
head = OutputHead(encoder.vocab_size)

# Load the saved state dictionary
print(f"Loading weights from {config.checkpoint_path}...")
checkpoint = torch.load(model_path, map_location=config.device, weights_only=True)
model.load_state_dict(checkpoint['model_state_dict'])
head.load_state_dict(checkpoint['head_state_dict'])

# Move it to the correct device (CPU, CUDA, or MPS)
m = model.to(config.device)
h = head.to(config.device)

m.eval() # Set model to evaluation mode
h.eval()

# Print parameters
total_params = sum(p.numel() for p in m.parameters()) + sum(p.numel() for p in h.parameters())
print(total_params/1e6, 'M parameters')

print("\n--- Generating some text ---")
# Start with a single blank token (0) as the context
context = torch.zeros((1, 1), dtype=torch.long, device=config.device)

# Generate and decode!
generated_tokens = h.generate(m, context, max_new_tokens=500)[0].tolist()
print(encoder.decode(generated_tokens))
