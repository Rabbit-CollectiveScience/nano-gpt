import torch
import torch.nn as nn
from torch.nn import functional as F
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
import config

def apply_rotary_emb(q, k):
    B, T, C = q.shape
    
    # 1. Compute frequencies
    pos = torch.arange(T, device=q.device).float()
    dim_arr = torch.arange(0, C, 2, device=q.device).float()
    freqs = 10000.0 ** (-dim_arr / C)
    
    # Outer product: (T, C//2)
    theta = torch.outer(pos, freqs)
    
    # 2. Get sin and cos, duplicate for both halves
    cos = torch.cos(theta) # (T, C//2)
    sin = torch.sin(theta)
    
    cos = torch.cat([cos, cos], dim=-1).unsqueeze(0) # (1, T, C)
    sin = torch.cat([sin, sin], dim=-1).unsqueeze(0) # (1, T, C)
    
    # 3. Rotate Q
    q_half1, q_half2 = q[..., :C//2], q[..., C//2:]
    q_rot = torch.cat([-q_half2, q_half1], dim=-1)
    q_out = q * cos + q_rot * sin
    
    # 4. Rotate K
    k_half1, k_half2 = k[..., :C//2], k[..., C//2:]
    k_rot = torch.cat([-k_half2, k_half1], dim=-1)
    k_out = k * cos + k_rot * sin
    
    return q_out, k_out

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(config.n_embd, head_size, bias=False)
        self.query = nn.Linear(config.n_embd, head_size, bias=False)
        self.value = nn.Linear(config.n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(config.block_size, config.block_size)))

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)   # (B,T,16)
        q = self.query(x) # (B,T,16)
        
        # Apply LLaMA-style Rotary Position Embeddings (RoPE)
        q, k = apply_rotary_emb(q, k)
        
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * C**-0.5 # (B, T, 16) @ (B, 16, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,16)
        out = wei @ v # (B, T, T) @ (B, T, 16) -> (B, T, 16)
        return out
