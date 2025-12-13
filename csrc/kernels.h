#pragma once
#include <torch/extension.h>

torch::Tensor fused_feed_forward_cpu(
    torch::Tensor input,
    torch::Tensor w1,
    torch::Tensor b1,
    torch::Tensor w2,
    torch::Tensor b2);

torch::Tensor tiled_attention_cpu(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    float scale);

torch::Tensor rms_norm_cpu(
    torch::Tensor x,
    torch::Tensor weight,
    float epsilon);

torch::Tensor apply_rope_cpu(
    torch::Tensor x,
    torch::Tensor cos,
    torch::Tensor sin);
