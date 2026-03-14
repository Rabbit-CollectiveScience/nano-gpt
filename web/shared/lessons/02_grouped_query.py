import torch
import torch.nn.functional as F
import math

# --- LESSON 2: Grouped-Query Attention (Mistral) ---
# Modern models like Mistral and LLaMA-3 use GQA.
# Watch what happens when we only use 1 Key matrix but 4 Query matrices! 
# We have to mathematically duplicate the Keys to make the tensor shapes fit.

n_heads = 4
n_kv_heads = 1 # Mistral groups 4 Queries to share just 1 Key!
head_dim = 16
T = 8 

torch.manual_seed(1337)
q = torch.randn(1, n_heads, T, head_dim)
k = torch.randn(1, n_kv_heads, T, head_dim)

# ⚠️ We mathematically duplicate the 1 Key into 4 clones so the matrices match!
n_rep = n_heads // n_kv_heads
k_expanded = torch.repeat_interleave(k, n_rep, dim=1)

# Now we can safely run the Attention Formula
scores = torch.matmul(q, k_expanded.transpose(-2, -1)) / math.sqrt(head_dim)

tril = torch.tril(torch.ones(T, T))
scores = scores.masked_fill(tril == 0, float('-inf'))
percentages = F.softmax(scores, dim=-1)

# Hand it off to the D3 Visualizer!
visualize_data = percentages[0].detach()
