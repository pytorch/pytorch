#pragma once

#include <ATen/core/Tensor.h>

namespace c10d::fsdp {

TORCH_API void check_split_with_sizes_copy_inputs(
    at::TensorList out,
    const at::Tensor& input,
    at::IntArrayRef split_sizes,
    at::IntArrayRef num_prefixes,
    int64_t num_chunks);

TORCH_API void check_chunk_cat_inputs(
    const at::Tensor& out,
    at::TensorList tensors,
    at::IntArrayRef num_leading_dims,
    int64_t num_chunks);

TORCH_API void split_with_sizes_copy_with_prefixes(
    at::TensorList out,
    const at::Tensor& input,
    at::IntArrayRef split_sizes,
    at::IntArrayRef num_prefixes,
    int64_t num_chunks);

TORCH_API at::Tensor& chunk_cat_with_prefixes(
    at::Tensor& out,
    at::TensorList tensors,
    at::IntArrayRef num_leading_dims,
    int64_t num_chunks);

} // namespace c10d::fsdp
