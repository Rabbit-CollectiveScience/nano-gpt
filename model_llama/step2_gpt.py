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
from model_llama.step2a_block import Block, RMSNorm

class GPTLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, config.n_embd)
        self.blocks = nn.Sequential(*[Block(config.n_embd, n_head=config.n_head) for _ in range(config.n_layer)])
        self.ln_f = RMSNorm(config.n_embd) # final layer norm

        # better init, not covered in the original Karpathy video but good practice
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        x = tok_emb # position information is now injected dynamically via RoPE in the attention heads
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        
        # Step 2 simply returns the pure mathematical context vectors
        return x
