import torch
import torch.nn as nn
import time
import math
from hypertext.ops import fused_feed_forward, tiled_attention, HAS_C_EXT
from hypertext.models.llama_proto import LlamaProto

def measure_flops(fn, *args, **kwargs):
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start = time.time()
    res = fn(*args, **kwargs)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    end = time.time()
    return res, end - start

def benchmark_attention():
    print(f"[{'C++' if HAS_C_EXT else 'PYTHON'}] Benchmarking Attention...")
    B, H, N, D = 4, 8, 1024, 64
    q = torch.randn(B, H, N, D)
    k = torch.randn(B, H, N, D)
    v = torch.randn(B, H, N, D)
    
    # Warmup
    for _ in range(5): tiled_attention(q, k, v)
    
    # Bench
    start = time.time()
    for _ in range(20):
        tiled_attention(q, k, v)
    dur = time.time() - start
    
    # Estimate FLOPs: 4 * B * H * N^2 * D
    flops = 4 * B * H * (N**2) * D * 20
    gflops = (flops / dur) / 1e9
    
    print(f"  > Avg Time: {dur/20*1000:.2f} ms | Throughput: {gflops:.2f} GFLOPs")
    return gflops

def benchmark_llama_inference():
    print(f"[{'C++' if HAS_C_EXT else 'PYTHON'}] Benchmarking LLaMA-Proto Inference...")
    vocab = 10000
    dim = 512
    layers = 4
    model = LlamaProto(vocab, dim, layers, 8)
    x = torch.randint(0, vocab, (1, 32))
    
    # Warmup
    model(x)
    
    start = time.time()
    iters = 50
    for _ in range(iters):
        model(x)
    dur = time.time() - start
    
    tps = (iters * 32) / dur
    print(f"  > Avg Time: {dur/iters*1000:.2f} ms | Tokens/Sec: {tps:.2f}")

def run_suite():
    print("="*60)
    print("  HyperText-Infinite Performance Studio")
    print(f"  Backend: {'Optimized C++ Kernels' if HAS_C_EXT else 'Pure Python Fallback (Slow)'}")
    print("="*60)
    
    benchmark_attention()
    benchmark_llama_inference()
    
    print("\n" + "="*60)
    print("  Report Generated Successfully.")
    print("="*60)

if __name__ == "__main__":
    run_suite()
