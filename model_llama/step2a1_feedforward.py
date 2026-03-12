import torch.nn as nn
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
import config

import torch.nn.functional as F

class FeedForward(nn.Module):
    """ a LLaMA-style SwiGLU FeedForward layer """

    def __init__(self, n_embd):
        super().__init__()
        # In LLaMA, the hidden dimension is roughly 8/3 * n_embd
        # For simplicity in our nano model, we'll keep a similar parameter count to 4*n_embd
        hidden_dim = int(8 * n_embd / 3) 
        
        # SwiGLU requires TWO parallel linear layers instead of one
        self.w1 = nn.Linear(n_embd, hidden_dim, bias=False) # The "Gate"
        self.w3 = nn.Linear(n_embd, hidden_dim, bias=False) # The "Value"
        
        # The output projection
        self.w2 = nn.Linear(hidden_dim, n_embd, bias=False)
        
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        # SwiGLU Math: (Swish(x * W1) * (x * W3)) * W2
        # F.silu is PyTorch's optimized version of the Swish activation function
        gate = F.silu(self.w1(x))
        value = self.w3(x)
        
        hidden = gate * value # This is the "Gating" mechanism
        
        out = self.w2(hidden)
        return self.dropout(out)
