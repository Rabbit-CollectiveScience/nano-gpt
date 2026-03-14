import torch
import torch.nn as nn
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

import config
from model_mixtral.step2a1_expert import FeedForward
from model_mixtral.step2a1b_router import TopKRouter

class SparseMoE(nn.Module):
    """ The master container that holds 1 Router and N Experts """
    def __init__(self, n_embd, n_experts, top_k):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        
        # Instantiate 1 Traffic Cop
        self.router = TopKRouter(n_embd, n_experts, top_k)
        
        # Instantiate 4 Identical SwiGLU Experts (ModuleList holds them side-by-side)
        self.experts = nn.ModuleList([FeedForward(n_embd) for _ in range(self.n_experts)])
        
    def forward(self, x):
        # x shape: [Batch, Time, n_embd]
        b, t, e = x.shape
        
        # 1. Ask the router who should handle these words
        # routing_weights: [B, T, top_k] (The confidence percentages)
        # routing_indices: [B, T, top_k] (The Expert IDs, e.g., 0 and 3)
        routing_weights, routing_indices = self.router(x)
        
        # Initialize an empty array of 0s to hold the final answers
        final_output = torch.zeros_like(x)
        
        # 2. Flatten out the Batch and Time dimensions to process every word individually
        # This makes it easier to assign specific words to specific experts
        flat_x = x.view(-1, e)
        flat_weights = routing_weights.view(-1, self.top_k)
        flat_indices = routing_indices.view(-1, self.top_k)
        flat_output = torch.zeros_like(flat_x)
        
        # 3. Route the data! 
        # For each of the Top-K slots (e.g. Slot 1 and Slot 2)
        for i in range(self.top_k):
            # Find out which Expert won this slot for every single word
            expert_ids_for_this_slot = flat_indices[:, i]
            expert_weights_for_this_slot = flat_weights[:, i]
            
            # Now, for every single physical Expert (0, 1, 2, and 3):
            for expert_idx, expert in enumerate(self.experts):
                # Find the words that were assigned to THIS specific expert
                # The _ says "I don't care about the true/false list, just give me the raw positions"
                _, word_positions = torch.where(expert_ids_for_this_slot == expert_idx)
                
                # If anyone was assigned to this expert...
                if len(word_positions) > 0:
                    # 3a. Extract only the words this expert is supposed to read
                    tokens_for_expert = flat_x[word_positions]
                    
                    # 3b. Pass them through the SwiGLU math!
                    expert_output = expert(tokens_for_expert)
                    
                    # 3c. Math trick: Multiply the output by the Router's confidence percentage
                    # (e.g., if the Router was only 20% sure, shrink the vector's impact by 80%)
                    expert_weight = expert_weights_for_this_slot[word_positions].unsqueeze(1)
                    weighted_output = expert_output * expert_weight
                    
                    # 3d. Add the newly refined vector back into its original slot in the sequence!
                    flat_output[word_positions] += weighted_output
                    
        # 4. Re-fold the array back into standard [Batch, Time, Embedding] shapes
        final_output = flat_output.view(b, t, e)
        return final_output
