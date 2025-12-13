import torch
import torch.nn as nn
from .ops import fused_feed_forward

class FusedFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w1 = nn.Parameter(torch.randn(d_ff, d_model) * 0.02)
        self.b1 = nn.Parameter(torch.zeros(d_ff))
        self.w2 = nn.Parameter(torch.randn(d_model, d_ff) * 0.02)
        self.b2 = nn.Parameter(torch.zeros(d_model))
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x is (Batch, Seq, Dim) or (Batch, Dim)
        # The C++ op expects (Batch, InputDim), so we might need to flatten
        is_3d = x.dim() == 3
        if is_3d:
            b, s, d = x.shape
            x_reshaped = x.view(b * s, d)
        else:
            x_reshaped = x
            
        output = fused_feed_forward(x_reshaped, self.w1, self.b1, self.w2, self.b2)
        
        if is_3d:
            output = output.view(b, s, -1)
            
        return self.dropout(output)

class HyperTransformerBlock(nn.Module):
    def __init__(self, d_model, heads, d_ff, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, heads, dropout=dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FusedFeedForward(d_model, d_ff, dropout)
        
    def forward(self, x, mask=None):
        # MHA
        attn_out, _ = self.attn(x, x, x, attn_mask=mask)
        x = self.ln1(x + attn_out)
        
        # Fused FFN
        ff_out = self.ffn(x)
        x = self.ln2(x + ff_out)
        return x
