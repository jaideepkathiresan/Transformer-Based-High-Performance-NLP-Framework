#include <torch/extension.h>
#include <vector>
#include <cmath>
#include <omp.h>

// Apply RoPE
// x: (Batch, Seq, Head, Dim) or (Batch, Seq, Dim) - We assume Dim is last
// cos: (Seq, Dim/2)
// sin: (Seq, Dim/2)
// This is a simplified application assuming 'cos' and 'sin' are precomputed and passed in.
torch::Tensor apply_rope_cpu(
    torch::Tensor x,
    torch::Tensor cos,
    torch::Tensor sin) {
    
    // Auto broadcast check omitted for brevity in prototype but essential in prod.
    // Assume x is (..., Seq, Dim)
    
    auto output = torch::empty_like(x);
    
    auto x_ptr = x.data_ptr<float>();
    auto c_ptr = cos.data_ptr<float>();
    auto s_ptr = sin.data_ptr<float>();
    auto out_ptr = output.data_ptr<float>();
    
    auto numel = x.numel();
    auto D = x.size(-1);
    auto N = numel / D; // Number of tokens * heads
    auto S = x.size(-2); // Sequence length dimension index varies, assuming cos/sin match logical seq index
    
    // Strictly, aligning the correct cos/sin index to x index is complex if shapes differ (broadcasting).
    // Let's assume x is flattened to (TotalTokens, Dim) and cos/sin are (TotalTokens, Dim/2) for simplicity 
    // OR we iterate specifically.
    // To be safer and robust for the demo:
    // Assume x: (Batch, SeqLen, Heads, HeadDim) -> This is standard LLaMA layout.
    // cos/sin: (SeqLen, HeadDim/2)
    // We need to iterate carefully.
    
    // If we assume the input is reshaped to allow simple indexing:
    
    int64_t B = x.size(0);
    int64_t SL = x.size(1);
    int64_t H = x.size(2);
    int64_t HD = x.size(3);
    
    // Parallelize over Batch, Seq, Head
    #pragma omp parallel for collapse(3)
    for (int b = 0; b < B; ++b) {
        for (int s = 0; s < SL; ++s) {
            for (int h = 0; h < H; ++h) {
                
                float* x_token = x_ptr + (b * SL * H * HD) + (s * H * HD) + (h * HD);
                float* out_token = out_ptr + (b * SL * H * HD) + (s * H * HD) + (h * HD);
                
                // Rotational Logic
                // Pairs (x[t], x[t+HD/2]) ? LLaMA typically interleaves or splits.
                // Standard RoPE: Split half. 
                // x_out[i] = x[i] * cos[s][i] - x[i + half] * sin[s][i]
                // x_out[i + half] = x[i] * sin[s][i] + x[i + half] * cos[s][i]
                
                int half = HD / 2;
                float* c_token = c_ptr + s * half;
                float* s_token = s_ptr + s * half;
                
                for (int i = 0; i < half; ++i) {
                    float x1 = x_token[i];
                    float x2 = x_token[i + half];
                    float c = c_token[i];
                    float s_val = s_token[i];
                    
                    out_token[i]        = x1 * c - x2 * s_val;
                    out_token[i + half] = x1 * s_val + x2 * c;
                }
            }
        }
    }
    
    return output;
}
