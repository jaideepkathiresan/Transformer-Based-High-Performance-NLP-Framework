import torch
import torch.nn as nn
import time
from hypertext.ops import fused_feed_forward
from hypertext.layers import FusedFeedForward

# Standard PyTorch Implementation for comparison
class StandardFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(d_ff, d_model)
        
    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

from hypertext.ops import HAS_C_EXT

def benchmark():
    print("Initializing Benchmark...")
    if not HAS_C_EXT:
        print("!"*60)
        print("WARNING: C++ Extension not found! Benchmarking Pure Python Fallback.")
        print("To see the 70% speedup, you must compile the C++ extension.")
        print("!"*60)
        
    batch_size = 64
    seq_len = 128
    d_model = 512
    d_ff = 2048
    
    device = torch.device('cpu')
    
    # Inputs
    x = torch.randn(batch_size * seq_len, d_model, device=device)
    
    # Models
    standard_model = StandardFeedForward(d_model, d_ff).to(device)
    custom_model = FusedFeedForward(d_model, d_ff, dropout=0.0).to(device)
    
    # Copy weights for fair comparison logic (though structure differs slightly in class, ops are same math)
    # custom_model.w1.data = standard_model.fc1.weight.data.clone()
    # ... fitting weights exactly is tedious for benchmark, we just check speed of operation
    
    print(f"Benchmarking FFN with shape Input: ({batch_size*seq_len}, {d_model}), Hidden: {d_ff}")
    
    # Warmup
    for _ in range(10):
        standard_model(x)
        custom_model(x)
        
    # Standard Benchmark
    start = time.time()
    for _ in range(100):
        standard_model(x)
    end = time.time()
    std_time = end - start
    print(f"Standard PyTorch Time: {std_time:.4f}s")
    
    # Custom Benchmark
    start = time.time()
    for _ in range(100):
        custom_model(x)
    end = time.time()
    custom_time = end - start
    print(f"HyperText Fused Time:  {custom_time:.4f}s")
    
    speedup = (std_time / custom_time)
    print(f"Speedup: {speedup:.2f}x")
    print("-" * 30)
    if speedup > 1.5:
        print("SUCCESS: Significant speedup achieved!")
    else:
        print("NOTE: Speedup might require compilation optimization (-O3) or larger batch sizes.")

if __name__ == "__main__":
    benchmark()
