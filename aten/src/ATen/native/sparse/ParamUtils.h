#pragma once

#include <ATen/core/Tensor.h>
#include <ATen/TensorUtils.h>
#include <tuple>

namespace at::native {

TORCH_API std::tuple<Tensor, Tensor, int64_t> softmax_sparse_input_preprocessing(
    const Tensor& input_,
    const int64_t dim_,
    const bool half_to_float,
    CheckedFrom function_name);

TORCH_API std::tuple<Tensor, Tensor, Tensor, int64_t> softmax_backward_sparse_input_preprocessing(
    const Tensor& grad_,
    const Tensor& output_,
    int64_t dim_,
    const Tensor& input_,
    CheckedFrom function_name);

} // namespace at::native
