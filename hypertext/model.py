import torch
import torch.nn as nn
from .layers import HyperTransformerBlock

class HyperBert(nn.Module):
    def __init__(self, vocab_size, d_model=512, n_layers=6, heads=8, d_ff=2048, max_len=512, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, max_len, d_model))
        self.layers = nn.ModuleList([
            HyperTransformerBlock(d_model, heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.fc_out = nn.Linear(d_model, vocab_size)
    
    def forward(self, x, mask=None):
        b, seq_len = x.shape
        x = self.embedding(x) + self.pos_emb[:, :seq_len, :]
        
        for layer in self.layers:
            x = layer(x, mask=mask)
            
        return self.fc_out(x)
