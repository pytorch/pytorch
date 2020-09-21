#pragma once

#include <ATen/ATen.h>
#include <ATen/TensorUtils.h>
#include <tuple>

namespace at {
namespace native {

namespace apply {

std::pair<Tensor, Tensor> softmax_sparse_input_preprocessing(
    const Tensor& input_,
    const int64_t dim_,
    const bool half_to_float,
    CheckedFrom function_name);

std::tuple<Tensor, Tensor, Tensor> softmax_backward_sparse_input_preprocessing(
    const Tensor& grad_,
    const Tensor& output_,
    int64_t dim_,
    const Tensor& input_,
    CheckedFrom function_name);

} // namespace apply

} // namespace native
} // namespace at
