import torch
from hypertext.models.llama_proto import LlamaProto
from hypertext.models.moe import MoELayer
import time

def demo_llama():
    print("="*60)
    print("  HyperText-Infinite: LLaMA-Proto Architecture Demo")
    print("="*60)
    
    # Config similar to Llama-2-7b scaled down
    vocab_size = 32000
    d_model = 1024
    n_layers = 4
    n_heads = 16 # Head dim 64
    
    print(f"Initializing LLaMA Model (Dim={d_model}, Layers={n_layers}, Heads={n_heads})...")
    model = LlamaProto(vocab_size, d_model, n_layers, n_heads)
    
    # Input
    batch_size = 4
    seq_len = 128
    x = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    print("Running Forward Pass (with RoPE + RMSNorm + FlashAttn)...")
    start = time.time()
    out = model(x)
    end = time.time()
    
    print(f"Output Shape: {out.shape}")
    print(f"Inference Time: {end - start:.4f}s")
    print("Success: LLaMA-Proto Pipeline Operational.")
    print("-" * 60)

def demo_moe():
    print("\n" + "="*60)
    print("  Mixture of Experts (MoE) Subsystem Demo")
    print("="*60)
    
    d_model = 1024
    d_ff = 4096
    num_experts = 8
    k = 2
    
    print(f"Initializing MoE Layer (Experts={num_experts}, TopK={k})...")
    moe = MoELayer(d_model, d_ff, num_experts, k)
    
    x = torch.randn(4, 32, d_model) # (Batch, Seq, Dim)
    
    print("Running Sparse Gating & Dispatch...")
    start = time.time()
    out = moe(x)
    end = time.time()
    
    print(f"Output Shape: {out.shape}")
    print(f"Execution Time: {end - start:.4f}s")
    print("Success: MoE Kernel Operational.")
    print("=" * 60)

if __name__ == "__main__":
    demo_llama()
    demo_moe()
