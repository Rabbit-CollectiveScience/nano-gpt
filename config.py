import torch

# Architecture Selection
# Options: 'gpt2' (Classic absolute embeddings) or 'llama' (Modern RoPE embeddings)
model_version = 'gpt2'

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
