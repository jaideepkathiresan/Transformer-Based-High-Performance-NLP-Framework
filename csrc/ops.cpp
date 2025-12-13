#include <torch/extension.h>
#include "kernels.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_feed_forward", &fused_feed_forward_cpu, "Fused Feed Forward (CPU)");
    m.def("tiled_attention", &tiled_attention_cpu, "Tiled Flash Attention (CPU)");
    m.def("rms_norm", &rms_norm_cpu, "RMS Normalization (CPU)");
    m.def("apply_rope", &apply_rope_cpu, "Rotary Positional Embeddings (CPU)");
}

