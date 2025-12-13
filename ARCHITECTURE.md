# HyperText-Infinite System Architecture

![System Architecture](PyTorch%20LLaMA%20MoE%20Training.png)

## System Components

### 1. Data Ingestion Layer (`hypertext/data`)
*   **Design**: Zero-copy Memory Mapping (`np.memmap`)
*   **Purpose**: Handle datasets larger than RAM (e.g., CommonCrawl) by paging data from SSD on demand.
*   **Optimization**: Prefetching and pinned memory in `DataLoader` for non-blocking GPU transfer.

### 2. Modeling Layer (`hypertext/models`)
*   **LLaMA Architecture**:
    *   **Pre-Normalization**: Stabilizes gradients in deep networks.
    *   **SwiGLU**: Gated Linear Units for better representation than ReLU.
    *   **RoPE**: Relative positional encoding for long-context generalization.
*   **Mixture of Experts (MoE)**:
    *   **Sparse Gating**: Only top-k experts are activated per token, keeping FLOPs constant while scaling parameters.

### 3. Distributed Parallelism (`hypertext/distributed`)
*   **Tensor Parallelism (TP)**:
    *   Splits individual weight matrices (e.g., $ W_Q, W_K, W_V $) across GPUS.
    *   **Column Parallel**: Splits output dimension (Linear 1).
    *   **Row Parallel**: Splits input dimension (Linear 2).
    *   Requires `AllReduce` synchronization in the forward pass.

### 4. Kernel Backend (`csrc/`)
*   **Latency Optimization**: Fusing operations (Linear+ReLU+Linear) reduces Main Memory Access (HBM/DRAM traffic), which is the primary bottleneck in Transformers.
*   **Cache Tiling**: The Flash Attention kernel processes blocks of $ Q, K, V $ that fit in L1/L2 cache, reducing cache misses by orders of magnitude.
