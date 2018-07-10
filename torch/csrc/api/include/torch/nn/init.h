#pragma once

#include <torch/tensor.h>

namespace torch {
namespace nn {
namespace init {

Tensor constant_(Tensor tensor, Scalar value);
Tensor dirac_(Tensor tensor);
Tensor eye_(Tensor tensor);
Tensor normal_(Tensor tensor, double mean = 0, double std = 1);
Tensor ones_(Tensor tensor);
Tensor orthogonal_(Tensor tensor, double gain = 1.0);
Tensor sparse_(Tensor tensor, double sparsity, double std = 0.01);
Tensor uniform_(Tensor tensor, double low = 0, double high = 1);
Tensor xavier_normal_(Tensor tensor, double gain = 1.0);
Tensor xavier_uniform_(Tensor tensor, double gain = 1.0);
Tensor zeros_(Tensor tensor);

} // namespace init
} // namespace nn
} // namespace torch
