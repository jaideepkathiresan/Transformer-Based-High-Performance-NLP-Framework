import torch
import torch.nn as nn
from torch.utils.cpp_extension import load

# Try to import the compiled extension
try:
    import hypertext._C as _C
    HAS_C_EXT = True
except ImportError:
    HAS_C_EXT = False
    print("Warning: C++ extension not found. Using slow pure-Python fallback.")

def fused_feed_forward(input, w1, b1, w2, b2):
    """
    Fused Feed Forward Network: Linear -> ReLU -> Linear
    
    Args:
        input: (batch, input_dim)
        w1: (hidden_dim, input_dim)
        b1: (hidden_dim)
        w2: (output_dim, hidden_dim)
        b2: (output_dim)
    """
    if HAS_C_EXT and input.device.type == 'cpu':
        return _C.fused_feed_forward(input, w1, b1, w2, b2)
    else:
        # Fallback for GPU or if C++ ext is missing
        # x @ w1.t() + b1
        x = torch.nn.functional.linear(input, w1, b1)
        x = torch.nn.functional.relu(x)
        x = torch.nn.functional.linear(x, w2, b2)
        return x

def tiled_attention(query, key, value, scale=None):
    """
    Tiled 'Flash' Attention using C++ kernels.
    """
    if scale is None:
        scale = 1.0 / (query.size(-1) ** 0.5)
        
    if HAS_C_EXT and query.device.type == 'cpu':
        return _C.tiled_attention(query, key, value, scale)
    else:
        # Fallback: Standard SDPA
        # This acts as a reference implementation
        return torch.nn.functional.scaled_dot_product_attention(
            query, key, value, scale=scale
        )

def rms_norm(x, weight, epsilon=1e-6):
    """
    Root Mean Square Normalization.
    """
    if HAS_C_EXT and x.device.type == 'cpu':
        return _C.rms_norm(x, weight, epsilon)
    else:
        # Fallback
        var = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(var + epsilon) * weight

def apply_rope(x, cos, sin):
    """
    Apply Rotary Positional Embeddings.
    """
    if HAS_C_EXT and x.device.type == 'cpu':
        return _C.apply_rope(x, cos, sin)
    else:
        # Fallback: Pure Python RoPE
        # x: (..., D)
        # Split into even/odd
        # x_out[2i]   = x[2i]*cos[i] - x[2i+1]*sin[i]
        # x_out[2i+1] = x[2i]*sin[i] + x[2i+1]*cos[i]
        
        # Reshape to pair
        d = x.shape[-1]
        x_reshaped = x.view(*x.shape[:-1], -1, 2)
        x1 = x_reshaped[..., 0]
        x2 = x_reshaped[..., 1]
        
        # Flatten cos/sin to match if needed, but assuming pre-broadcasted or matching seq len
        # Simplified:
        out1 = x1 * cos - x2 * sin
        out2 = x1 * sin + x2 * cos
        
        return torch.stack([out1, out2], dim=-1).flatten(-2)


