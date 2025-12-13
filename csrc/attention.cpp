#include <torch/extension.h>
#include <vector>
#include <cmath>
#include <omp.h>
#include <algorithm>

// Tiled "Flash" Attention for CPU
// This is a simplified demonstration of tiling to fit blocks in L2 Cache.
// Logic:
// 1. Iterate over blocks of Q (Output blocks)
// 2. Iterate over blocks of K/V (Input blocks)
// 3. Compute Attention Scores -> Softmax -> V aggregation
// 4. Use online softmax rescaling (Safe Softmax)

torch::Tensor tiled_attention_cpu(
    torch::Tensor query, // (Batch, Heads, SeqLen, Dim)
    torch::Tensor key,   // (Batch, Heads, SeqLen, Dim)
    torch::Tensor value, // (Batch, Heads, SeqLen, Dim)
    float scale) {
    
    // Checks (simplified)
    TORCH_CHECK(query.device().is_cpu(), "Query must be on CPU");
    
    auto B = query.size(0);
    auto H = query.size(1);
    auto N = query.size(2); // Sequence Length
    auto D = query.size(3); // Head Dimension
    
    auto output = torch::zeros_like(query);
    
    // Accessors
    auto q_ptr = query.data_ptr<float>();
    auto k_ptr = key.data_ptr<float>();
    auto v_ptr = value.data_ptr<float>();
    auto out_ptr = output.data_ptr<float>();
    
    // Block Sizes - Tuned for L2/L3 Cache lines
    // For a real Microsoft-level project, these would be auto-tuned or heuristic-based.
    const int Br = 64; // Block size for rows (Query)
    const int Bc = 64; // Block size for cols (Key/Value)
    
    // Parallelize over Batch and Heads
    #pragma omp parallel for collapse(2)
    for (int b = 0; b < B; ++b) {
        for (int h = 0; h < H; ++h) {
            
            // Base pointers for this batch/head
            long offset = b * (H * N * D) + h * (N * D);
            float* Q_local = q_ptr + offset;
            float* K_local = k_ptr + offset;
            float* V_local = v_ptr + offset;
            float* O_local = out_ptr + offset;
            
            // Temporary buffers for online softmax (Rows)
            // L_i = sum(exp(S_ij))
            // M_i = max(S_ij)
            std::vector<float> L(N, 0.0f);
            std::vector<float> M(N, -1e9f); // neg infinity
            
            // Tiling Loops
            // Iterate over Key/Value blocks (Outer loop in FlashAttn V2 style for some variants, 
            // but standard tiling often does Q then K. Let's do standard tiling)
            
            for (int i = 0; i < N; i += Br) {
                int valid_Br = std::min(Br, (int)(N - i));
                
                for (int j = 0; j < N; j += Bc) {
                    int valid_Bc = std::min(Bc, (int)(N - j));
                    
                    // Process Block (Br x Bc)
                    for (int row = 0; row < valid_Br; ++row) {
                        int global_row = i + row;
                        float max_val = M[global_row];
                        float sum_exp = L[global_row];
                        
                        // We accumulate updates to Output in registers/cache
                        // But since we need to re-scale previous outputs when Max changes, 
                        // we need careful handling. 
                        // Simplified approach: Re-read O_local[global_row], update, write back.
                        
                        // 1. Compute Scores Q[row] dot K[col_range]
                        for (int col = 0; col < valid_Bc; ++col) {
                            int global_col = j + col;
                            
                            float score = 0.0f;
                            for (int d = 0; d < D; ++d) {
                                score += Q_local[global_row * D + d] * K_local[global_col * D + d];
                            }
                            score *= scale;
                            
                            // 2. Online Softmax update logic
                            float old_max = max_val;
                            if (score > max_val) {
                                max_val = score;
                            }
                            
                            // Re-scale factor for existing accumulation
                            float rescale = std::exp(old_max - max_val);
                            // Exponent of current score
                            float exp_score = std::exp(score - max_val);
                            
                            // Update running sum
                            sum_exp = sum_exp * rescale + exp_score;
                            
                            // Update Output: O = O * rescale + V * exp_score
                            // Note: This is an inner loop over D, might be heavy.
                            // In real kernels, this is vectorized.
                            for (int d = 0; d < D; ++d) {
                                O_local[global_row * D + d] *= rescale;
                                O_local[global_row * D + d] += V_local[global_col * D + d] * exp_score;
                            }
                        }
                        
                        // Save statistics back
                        M[global_row] = max_val;
                        L[global_row] = sum_exp;
                    }
                }
            }
            
            // Final normalization: O = O / L
            for (int i = 0; i < N; ++i) {
                if (L[i] > 0) {
                    for (int d = 0; d < D; ++d) {
                        O_local[i * D + d] /= L[i];
                    }
                }
            }
        }
    }
    
    return output;
}

// Binding
torch::Tensor rms_norm_cpu(torch::Tensor x, torch::Tensor weight, float epsilon);

void init_kernels(pybind11::module& m) {
    m.def("tiled_attention", &tiled_attention_cpu, "Tiled Attention (CPU)");
    // We will define rms_norm in a separate file or merge here. 
    // For simplicity of the huge file write, let's keep them separate in C++ but bind here? 
    // Actually, Pybind needs one entry point usually or linked together.
    // I will use a central update to `ops.cpp` that includes these.
}
