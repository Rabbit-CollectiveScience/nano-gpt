import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
import config

def apply_rotary_emb(q, k):
    """
    LLaMA / Mistral Rotary Position Embeddings (RoPE)
    """
    B, T, num_q_heads, head_dim = q.shape
    _, _, num_kv_heads, _ = k.shape # Note: K might have fewer heads!
    
    # 1. Compute frequencies
    pos = torch.arange(T, device=q.device).float()
    dim_arr = torch.arange(0, head_dim, 2, device=q.device).float()
    freqs = 10000.0 ** (-dim_arr / head_dim)
    
    # Outer product: (T, head_dim // 2)
    theta = torch.outer(pos, freqs)
    
    # 2. Get sin and cos, duplicate for both halves
    cos = torch.cos(theta) # (T, head_dim // 2)
    sin = torch.sin(theta)
    
    cos = torch.cat([cos, cos], dim=-1).unsqueeze(0).unsqueeze(2) # (1, T, 1, head_dim)
    sin = torch.cat([sin, sin], dim=-1).unsqueeze(0).unsqueeze(2) # (1, T, 1, head_dim)
    
    # 3. Rotate Q
    q_half1, q_half2 = q[..., :head_dim//2], q[..., head_dim//2:]
    q_rot = torch.cat([-q_half2, q_half1], dim=-1)
    q_out = q * cos + q_rot * sin
    
    # 4. Rotate K (accounting for varying KV shapes)
    k_half1, k_half2 = k[..., :head_dim//2], k[..., head_dim//2:]
    k_rot = torch.cat([-k_half2, k_half1], dim=-1)
    k_out = k * cos + k_rot * sin
    
    return q_out, k_out

class MultiHeadAttention(nn.Module):
    """ 
    Grouped-Query Attention (GQA) used by Mistral & LLaMA-3.
    Instead of projecting N isolated Heads, we mathematically process the entire 
    tensor layer simultaneously. This allows N Query Heads to share 1 Key/Value array.
    """
    def __init__(self, n_head, head_size):
        super().__init__()
        self.n_head = n_head
        self.n_kv_head = config.n_kv_heads
        
        # In GQA, we divide Q Heads by KV Heads to see how many Qs share a KV
        self.n_rep = self.n_head // self.n_kv_head
        self.head_dim = head_size
        
        # The Projections
        # Queries still project to the full size (e.g. 32 heads * 128 dim)
        self.wq = nn.Linear(config.n_embd, self.n_head * self.head_dim, bias=False)
        
        # Keys and Values project to a SMALLER size! (e.g. 8 heads * 128 dim)
        self.wk = nn.Linear(config.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.wv = nn.Linear(config.n_embd, self.n_kv_head * self.head_dim, bias=False)
        
        self.wo = nn.Linear(config.n_head * self.head_dim, config.n_embd, bias=False)
        
        # Trillium mask for causal autoregression
        self.register_buffer('tril', torch.tril(torch.ones(config.block_size, config.block_size)))
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        B, T, C = x.shape
        
        # 1. Linear Projections
        xq = self.wq(x) # [B, T, n_head * head_dim]
        xk = self.wk(x) # [B, T, n_kv_head * head_dim]
        xv = self.wv(x) # [B, T, n_kv_head * head_dim]
        
        # 2. Reshape to split the flat tensor into distinct independent heads
        # Shape becomes: [Batch, Time, number_of_heads, dimension_of_head]
        xq = xq.view(B, T, self.n_head, self.head_dim)
        xk = xk.view(B, T, self.n_kv_head, self.head_dim)
        xv = xv.view(B, T, self.n_kv_head, self.head_dim)
        
        # 3. Apply Rotary Position Embeddings (RoPE)
        # We pass them in as [B, T, Heads, Dim]. The RoPE function rotates the final Dimension.
        xq, xk = apply_rotary_emb(xq, xk)
        
        # 4. Math Trick: Grouped-Query Expansion!
        # If we have 32 Q Heads but only 8 KV Heads, we mathematically duplicate each K and V 
        # 4 times! This "un-compresses" them so we can do standard attention math.
        # torch.repeat_interleave turns [H1, H2] into [H1, H1, H2, H2]
        if self.n_kv_head != self.n_head:
            xk = torch.repeat_interleave(xk, self.n_rep, dim=2) # [B, T, n_head, head_dim]
            xv = torch.repeat_interleave(xv, self.n_rep, dim=2) # [B, T, n_head, head_dim]
            
        # 5. Transpose shapes from [Batch, Time, Head, Dim] to [Batch, Head, Time, Dim]
        # This groups the data by Head so they don't mathematically collide during the dot-product
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)
        
        # 6. The Standard Attention Formula: Softmax(Q @ K.T / sqrt(d)) @ V
        # Because we grouped by Head on dimension 1, Python does `n_head` parallel dot-products!
        scores = torch.matmul(xq, xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply the Causal Mask (can't see the future)
        scores = scores.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        
        # Convert raw scores to percentages (0.0 to 1.0)
        scores = F.softmax(scores, dim=-1)
        scores = self.dropout(scores)
        
        # Re-weight the Values according to the attention percentages
        output = torch.matmul(scores, xv) # [B, n_head, T, head_dim]
        
        # 7. Re-flatten the tensor! 
        # Move Time back to dimension 1: [B, T, n_head, head_dim]
        output = output.transpose(1, 2).contiguous()
        # Squash the last two dimensions together: [B, T, n_head * head_dim]
        output = output.view(B, T, -1)
        
        # 8. Final dense projection
        return self.wo(output)
