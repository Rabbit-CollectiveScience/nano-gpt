import torch

# Architecture Selection
# Choose which model to spin up: 'gpt2', 'llama', 'mixtral', or 'mistral'
model_version = 'mistral'

# --- Mistral (GQA) Specific Hyperparameters ---
n_kv_heads = 1 # Number of Key/Value heads (Must divide evenly into n_head)

# --- Mixtral (MoE) Specific Hyperparameters ---

# Hyperparameters based on standard character-level GPT
batch_size = 8 # shrunk to fit MPS memory
block_size = 64 # shrunk to fit MPS memory
max_iters = 200
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2

# Path to save/load model weights
checkpoint_path = 'nano_gpt.pt'
