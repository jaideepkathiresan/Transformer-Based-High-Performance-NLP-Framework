#include <torch/extension.h>
#include <vector>
#include <cmath>
#include <omp.h>

torch::Tensor rms_norm_cpu(torch::Tensor x, torch::Tensor weight, float epsilon) {
    // x: (Batch, Seq, Dim) or equivalent last dim
    auto output = torch::empty_like(x);
    
    auto x_ptr = x.data_ptr<float>();
    auto w_ptr = weight.data_ptr<float>();
    auto out_ptr = output.data_ptr<float>();
    
    auto numel = x.numel();
    auto D = x.size(-1); // Last dimension
    auto N = numel / D;  // Number of tokens/vectors
    
    #pragma omp parallel for
    for (int i = 0; i < N; ++i) {
        float* row_in = x_ptr + i * D;
        float* row_out = out_ptr + i * D;
        
        // Sum of squares
        float sum_sq = 0.0f;
        
        // SIMD Friendly Loop
        for (int j = 0; j < D; ++j) {
            sum_sq += row_in[j] * row_in[j];
        }
        
        float inv_rms = 1.0f / std::sqrt(sum_sq / D + epsilon);
        
        for (int j = 0; j < D; ++j) {
            row_out[j] = row_in[j] * inv_rms * w_ptr[j];
        }
    }
    
    return output;
}
