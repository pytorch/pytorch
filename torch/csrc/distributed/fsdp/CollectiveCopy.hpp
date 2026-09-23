#pragma once

#include <ATen/core/Tensor.h>
#include <vector>

namespace c10d::fsdp {

// Returns whether copying requires resizing views of cached outputs.
TORCH_API bool check_all_gather_copy_out_inputs(
    at::TensorList out,
    const at::Tensor& input,
    at::IntArrayRef split_sizes,
    at::IntArrayRef outer_sizes,
    int64_t num_chunks);

TORCH_API void check_reduce_scatter_copy_in_inputs(
    const at::Tensor& out,
    at::TensorList tensors,
    at::IntArrayRef num_leading_dims,
    int64_t num_chunks);

TORCH_API void all_gather_copy_out(
    at::TensorList out,
    const at::Tensor& input,
    at::IntArrayRef split_sizes,
    at::IntArrayRef outer_sizes,
    int64_t num_chunks);

TORCH_API std::vector<at::Tensor> split_all_gather_output_with_resize(
    at::TensorList out,
    const at::Tensor& input,
    at::IntArrayRef split_sizes,
    at::IntArrayRef outer_sizes,
    int64_t num_chunks);

TORCH_API at::Tensor& reduce_scatter_copy_in(
    at::Tensor& out,
    at::TensorList tensors,
    at::IntArrayRef num_leading_dims,
    int64_t num_chunks);

} // namespace c10d::fsdp
