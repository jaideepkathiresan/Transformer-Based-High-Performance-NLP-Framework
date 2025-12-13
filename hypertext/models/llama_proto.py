import torch
import torch.nn as nn
import math
from hypertext.ops import rms_norm, tiled_attention, apply_rope

class HyperRMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps
        
    def forward(self, x):
        return rms_norm(x, self.weight, self.eps)

class HyperRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=2048):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_seq_len).float()
        freqs = torch.einsum('i,j->ij', t, inv_freq)
        # Create cos/sin
        self.register_buffer('cos', freqs.cos())
        self.register_buffer('sin', freqs.sin())
        
    def forward(self, x, seq_len):
        return self.cos[:seq_len, :], self.sin[:seq_len, :]

class HyperAshAttention(nn.Module):
    """
    Flash Attention with RoPE
    """
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        
    def forward(self, x, rope_cos, rope_sin):
        B, Seq, D = x.shape
        
        q = self.q_proj(x).view(B, Seq, self.n_heads, self.d_head)
        k = self.k_proj(x).view(B, Seq, self.n_heads, self.d_head)
        v = self.v_proj(x).view(B, Seq, self.n_heads, self.d_head)
        
        # Apply RoPE (Needs reshaping for efficient broadcast or kernel support)
        # Our Kernel expects specific layout or broadcasting. 
        # Python Fallback expects (..., Dim/2) for cos/sin but we generated (Seq, Dim/2).
        # We need to unsqueeze heads for broadcast: (Seq, 1, Dim/2)
        
        rope_cos = rope_cos.unsqueeze(1) # (Seq, 1, HeadDim/2)
        rope_sin = rope_sin.unsqueeze(1)
        
        # We need to flatten heads into batch or keep 4D. 
        # The C++ kernel apply_rope handles 4D but simplisticly.
        # Let's use Python fallback logic mostly here unless compiled.
        
        # Reshape for RoPE: (B, Seq, Heads, D)
        q = apply_rope(q, rope_cos, rope_sin)
        k = apply_rope(k, rope_cos, rope_sin)
        
        # Flash Attention
        # Reshape to (B, Heads, Seq, D) for attention
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        output = tiled_attention(q, k, v)
        
        output = output.transpose(1, 2).reshape(B, Seq, D)
        return self.o_proj(output)

class LlamaProto(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, n_heads):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.rope = HyperRotaryEmbedding(d_model // n_heads)
        
        self.layers = nn.ModuleList([])
        for _ in range(n_layers):
            layer = nn.ModuleDict({
                'attn': HyperAshAttention(d_model, n_heads),
                'norm1': HyperRMSNorm(d_model),
                'ffn': nn.Sequential(
                    nn.Linear(d_model, 4*d_model),
                    nn.SiLU(),
                    nn.Linear(4*d_model, d_model)
                ), # Using standard SiLU FFN for LLaMA compat vs fused relu
                'norm2': HyperRMSNorm(d_model)
            })
            self.layers.append(layer)
            
        self.norm_out = HyperRMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
    def forward(self, idx):
        B, Seq = idx.shape
        h = self.embed(idx)
        cos, sin = self.rope(h, Seq)
        
        for layer in self.layers:
            # Pre-Norm
            normed = layer['norm1'](h)
            attn_out = layer['attn'](normed, cos, sin)
            h = h + attn_out
            
            normed = layer['norm2'](h)
            ff_out = layer['ffn'](normed)
            h = h + ff_out
            
        return self.lm_head(self.norm_out(h))
