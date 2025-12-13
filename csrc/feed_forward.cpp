#include <torch/extension.h>
#include <vector>
#include <omp.h>

// Fused Linear -> ReLU -> Linear
torch::Tensor fused_feed_forward_cpu(
    torch::Tensor input,
    torch::Tensor w1,
    torch::Tensor b1,
    torch::Tensor w2,
    torch::Tensor b2) {
    
    TORCH_CHECK(input.device().is_cpu(), "input must be a CPU tensor");
    
    const int64_t batch_size = input.size(0);
    const int64_t hidden_dim = w1.size(0); 
    const int64_t output_dim = w2.size(0); 
    
    auto output = torch::empty({batch_size, output_dim}, input.options());
    
    auto input_a = input.accessor<float, 2>();
    auto w1_a = w1.accessor<float, 2>();
    auto b1_a = b1.accessor<float, 1>();
    auto w2_a = w2.accessor<float, 2>();
    auto b2_a = b2.accessor<float, 1>();
    auto output_a = output.accessor<float, 2>();
    
    #pragma omp parallel for
    for (int i = 0; i < batch_size; ++i) {
        std::vector<float> hidden_val(hidden_dim);
        
        for (int h = 0; h < hidden_dim; ++h) {
            float val = b1_a[h];
            for (int k = 0; k < input.size(1); ++k) {
                val += input_a[i][k] * w1_a[h][k];
            }
            hidden_val[h] = val > 0 ? val : 0;
        }
        
        for (int o = 0; o < output_dim; ++o) {
            float val = b2_a[o];
            for (int h = 0; h < hidden_dim; ++h) {
                val += hidden_val[h] * w2_a[o][h];
            }
            output_a[i][o] = val;
        }
    }
    return output;
}
