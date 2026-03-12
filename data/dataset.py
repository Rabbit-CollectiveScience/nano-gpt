import torch
import os
import sys

# Get the directory of the current file (data/) and the parent directory (nano-gpt/)
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

# Add parent directory to sys.path so we can import config
sys.path.append(parent_dir)
import config

data_path = os.path.join(parent_dir, 'input.txt')

# Read the dataset
with open(data_path, 'r', encoding='utf-8') as f:
    text = f.read()

import tiktoken

# Load OpenAI's GPT-4 BPE tokenizer
enc = tiktoken.get_encoding("cl100k_base")

# The vocabulary size of this tokenizer is exactly 100,277
vocab_size = enc.n_vocab

# Encoder: takes a string, outputs a list of integers
encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
# Decoder: takes a list of integers, outputs a string
decode = lambda l: enc.decode(l)

# Train and validation splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    """
    Generate a small batch of data of inputs x and targets y
    """
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - config.block_size, (config.batch_size,))
    x = torch.stack([data[i:i+config.block_size] for i in ix])
    y = torch.stack([data[i+1:i+config.block_size+1] for i in ix])
    x, y = x.to(config.device), y.to(config.device)
    return x, y
