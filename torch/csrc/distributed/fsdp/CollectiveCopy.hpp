#pragma once

#include <ATen/core/Tensor.h>

namespace torch::distributed::fsdp {

// Composite kernels of fsdp::_all_gather_copy_out_ and
// fsdp::_reduce_scatter_copy_in_. They validate their inputs with the check
// functions, which fused backend kernels call themselves.
TORCH_API void check_all_gather_copy_out_inputs(
    at::TensorList out,
    const at::Tensor& input,
    c10::SymIntArrayRef split_sizes,
    c10::SymIntArrayRef outer_sizes,
    int64_t num_chunks);

TORCH_API void all_gather_copy_out(
    at::TensorList out,
    const at::Tensor& input,
    c10::SymIntArrayRef split_sizes,
    c10::SymIntArrayRef outer_sizes,
    int64_t num_chunks);

TORCH_API void check_reduce_scatter_copy_in_inputs(
    const at::Tensor& out,
    at::TensorList tensors,
    at::IntArrayRef num_leading_dims,
    int64_t num_chunks);

TORCH_API at::Tensor& reduce_scatter_copy_in(
    at::Tensor& out,
    at::TensorList tensors,
    at::IntArrayRef num_leading_dims,
    int64_t num_chunks);

} // namespace torch::distributed::fsdp
