#pragma once

#include <ATen/core/Tensor.h>
#include <optional>

void f8f8bf16_rowwise_sm9x(at::Tensor XQ, at::Tensor WQ, at::Tensor x_scale,
                           at::Tensor w_scale, std::optional<at::Tensor> bias,
                           bool use_fast_accum, at::Tensor& out);
