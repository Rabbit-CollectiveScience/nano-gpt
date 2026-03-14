import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
import config

class TopKRouter(nn.Module):
    """ 
    The "Traffic Cop" of the Mixture of Experts. 
    It reads a token's vector and decides which Experts are most qualified to process it.
    """
    def __init__(self, n_embd, n_experts, top_k):
        super().__init__()
        self.top_k = top_k
        # A simple linear layer that outputs 1 probability-score per Expert
        self.routing_weights = nn.Linear(n_embd, n_experts, bias=False)

    def forward(self, x):
        # 1. Calculate the raw score of how much each expert wants this token
        # x shape: [Batch, Time, n_embd]
        # output shape: [Batch, Time, n_experts]
        routing_logits = self.routing_weights(x)
        
        # 2. Convert raw scores into percentages (0% to 100%)
        routing_probs = F.softmax(routing_logits, dim=-1)
        
        # 3. Only keep the top K experts! (e.g., the top 2 percentages)
        # topk_weights will be [Batch, Time, top_k] (The actual percentages)
        # topk_indices will be [Batch, Time, top_k] (The ID of the winning experts: 0, 1, 2, or 3)
        topk_weights, topk_indices = torch.topk(routing_probs, self.top_k, dim=-1)
        
        # 4. Mathematically normalize the top-k winning weights so they still sum to 100%
        # (e.g. if Expert 2 won with 60% and Expert 4 got 20%, we boost them to 75% and 25%)
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        
        return topk_weights, topk_indices
