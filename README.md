# HyperText-Infinite: Enterprise-Grade NLP Framework

> **Status**: High-Performance Research Prototype
> **Focus**: System Optimization, Kernel Fusion, Hybrid Architecture

HyperText-Infinite is a next-generation NLP framework designed to push the boundaries of PyTorch by offloading critical Transformer operations to hand-written **C++ / OpenMP** kernels.

It implements modern Large Language Model (LLM) architectures (LLaMA, Mixture of Experts) from scratch, prioritizing inference latency and memory bandwidth efficiency.

## Key Features

### 1. Advanced Kernel Suite (`csrc/`)
We bypass standard operators for critical paths:
*   **Tiled Flash Attention**: `csrc/attention.cpp` - Implements blocking strategies to optimize CPU L2 cache usage.
*   **Rotary Positional Embeddings (RoPE)**: `csrc/rope.cpp` - On-the-fly complex number rotation for position encoding.
*   **RMSNorm**: `csrc/rms_norm.cpp` - SIMD-optimized normalization for deep networks.
*   **Fused Feed-Forward**: `csrc/feed_forward.cpp` - Linear-ReLU-Linear fusion to reduce memory reads/writes.

### 2. Next-Gen Architectures (`hypertext/models/`)
*   **LLaMA-Proto**: A faithful reproduction of the LLaMA architecture using Pre-Norm, RMSNorm, and RoPE.
*   **Sparse Mixture of Experts (MoE)**: `hypertext/models/moe.py` - Implements Top-2 Gating and expert routing, demonstrating capability for massive-model scaling.

### 3. Enterprise Training loop
*   **`HyperTrainer`**: Supports Gradient Accumulation, Automatic Mixed Precision (AMP), and Checkpointing.
*   **Quantization Utilities**: Simulations for Int8 inference flow.

## Technical Stack
*   **Languages**: Python 3.9+, C++17
*   **Parallelism**: OpenMP (CPU), CUDA Ready (Architecture designed for GPU porting)
*   **Build System**: Custom `torch.utils.cpp_extension` setup for multi-file compilation.

## Quick Start

### Installation
```bash
# Editable install (Recommended)
# Falls back to Pure Python if C++ compiler is missing
pip install -e .
```

### Running the Demos

**1. Architecture Scale Demo** (LLaMA + MoE)
```bash
python examples/scale_demo.py
```

**2. Inference Generation (Throughput Test)**
```bash
python examples/generate_demo.py
```

**3. Performance Benchmarks**
```bash
python benchmarks/benchmark_speed.py
```

## Design Philosophy
This project demonstrates:
1.  **Low-Level Systems Programming**: Ability to write custom extensions for PyTorch.
2.  **Algorithm Optimization**: Understanding of Cache Tiling and Kernel Fusion.
3.  **Modern AI Architectures**: Implementation of RoPE, SwiGLU, and MoE from papers, not just using HuggingFace `AutoModel`.
4.  **Software Engineering**: Modular design, separation of concerns (Ops vs Layers vs Models), and robust build tooling.

---
*Built using PyTorch & C++*
