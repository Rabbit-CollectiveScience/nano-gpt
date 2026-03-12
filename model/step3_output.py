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
from model.step1_tokenizer import encoder

class OutputHead(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.lm_head = nn.Linear(config.n_embd, vocab_size)
        
        # Initialize weights
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.02)
        if self.lm_head.bias is not None:
            torch.nn.init.zeros_(self.lm_head.bias)

    def forward(self, x, targets=None):
        # x is the output from the transformer blocks: (B, T, n_embd)
        logits = self.lm_head(x) # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    @torch.no_grad()
    def generate(self, model, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -config.block_size:]
            
            # 1. Pass through the full GPT model (which now only returns `x` vectors)
            x = model(idx_cond)
            
            # focus only on the last time step
            x = x[:, -1, :] # becomes (B, C)
            
            # 2. Pass through this Output Head to get logits
            logits = self.lm_head(x) # (B, vocab_size)
            
            # 3. Apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, vocab_size)
            
            # 4. Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
            
        return idx
