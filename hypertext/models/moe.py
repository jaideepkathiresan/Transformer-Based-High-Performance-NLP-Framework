import torch
import torch.nn as nn
import torch.nn.functional as F
from hypertext.layers import FusedFeedForward

class MoELayer(nn.Module):
    def __init__(self, d_model, d_ff, num_experts=8, k=2):
        super().__init__()
        self.num_experts = num_experts
        self.k = k
        self.router = nn.Linear(d_model, num_experts)
        
        # Experts: Using our highly optimized C++ Fused Kernels
        self.experts = nn.ModuleList([
            FusedFeedForward(d_model, d_ff, dropout=0.0) 
            for _ in range(num_experts)
        ])
        
    def forward(self, x):
        # x: (Batch, Seq, Dim)
        B, Seq, Dim = x.shape
        x_flat = x.view(-1, Dim)
        
        # Routing
        router_logits = self.router(x_flat)
        routing_weights, selected_experts = torch.topk(router_logits, self.k, dim=-1)
        routing_weights = F.softmax(routing_weights, dim=-1)
        
        # Mock Dispatch (Sequential loop for simplicity in proto, real would use scatter/gather)
        # For 'Microsoft Scale', we'd write a C++ kernel for this dispatch.
        # Here we simulate the effect.
        
        final_output = torch.zeros_like(x_flat)
        
        # Naive iteration over experts (optimization opportunity mentioned in docs)
        # In a real heavy implementation, we would group tokens by expert.
        for i in range(self.num_experts):
            # Find tokens assigned to this expert
            # shape of selected_experts: (Tokens, k)
            # Create mask: (Tokens, k) boolean
            mask = (selected_experts == i)
            # Any token that selected this expert in any of its k slots?
            token_mask = mask.any(dim=-1) # (Tokens,)
            
            if token_mask.any():
                # Extract tokens
                tokens_for_expert = x_flat[token_mask]
                
                # Run Expert (Fused C++ Kernel)
                expert_out = self.experts[i](tokens_for_expert)
                
                # Weighted combination
                # We need to broadcast the weight correctly
                # mask has shape (Tokens, k). find which k index matched 'i'
                # This is getting complex for a naive loop, simplified assumption:
                
                # Simplified MoE: Only support k=1 for this "naive" demo or just sum up
                # Correct way: scatter add.
                
                # Let's do a weighted add properly:
                # Get the weight corresponding to this expert
                # weights: (Tokens, k)
                # We need weights[token_idx, slot_idx] where selected_experts[token_idx, slot_idx] == i
                
                # Gather weights for these tokens
                relevant_weights = routing_weights[token_mask] # (Subset, k)
                relevant_indices = selected_experts[token_mask] # (Subset, k)
                
                # Mask inside the subset
                subset_mask = (relevant_indices == i) # (Subset, k)
                
                # Select weight (sum if multiple times? unlikely with topk)
                # (Subset, 1)
                gate_val = (relevant_weights * subset_mask.float()).sum(dim=1, keepdim=True)
                
                final_output[token_mask] += expert_out * gate_val
                
        return final_output.view(B, Seq, Dim)
