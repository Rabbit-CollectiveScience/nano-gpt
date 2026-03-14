import torch
import torch.nn.functional as F
import math

# --- LESSON 1: Classic Self-Attention (GPT-2) ---
# Here we have 4 independent Attention Heads.
# Notice how we mathematically isolate each head by allocating them their own slice of the Key and Value matrices.

n_heads = 4
head_dim = 16
T = 8  # Sequence Length

torch.manual_seed(1337)

# GPT-2 assigns 1 specific Query to 1 specific Key and Value 
q = torch.randn(1, n_heads, T, head_dim)
k = torch.randn(1, n_heads, T, head_dim)

# Calculate Attention Matrix
scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)

# Apply Causal Mask (can't cheat and look into the future!)
tril = torch.tril(torch.ones(T, T))
scores = scores.masked_fill(tril == 0, float('-inf'))

percentages = F.softmax(scores, dim=-1)

# Hand it off to the D3 Visualizer!
visualize_data = percentages[0].detach()
