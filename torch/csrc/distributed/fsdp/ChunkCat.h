#pragma once

#include <ATen/ATen.h>
#include <torch/csrc/Export.h>

namespace torch::distributed::fsdp {

// Composite kernel of fsdp::chunk_cat_mixed_dtype. It calls _chunk_cat.out,
// first casting each input to out's dtype if the input dtypes differ, and
// reordering each input with leading dims so that its chunks along dim 0 are
// the chunks of its rows.
TORCH_API void chunk_cat_mixed_dtype(
    at::TensorList tensors,
    int64_t dim,
    int64_t num_chunks,
    at::Tensor& out,
    at::OptionalIntArrayRef num_leading_dims);

// Validates fsdp::chunk_cat_mixed_dtype inputs with leading dims, which fused
// backend kernels call themselves.
TORCH_API void check_chunk_cat_leading_dims_inputs(
    at::TensorList tensors,
    at::IntArrayRef num_leading_dims,
    int64_t num_chunks,
    const at::Tensor& out);

// Whether any input of fsdp::chunk_cat_mixed_dtype has leading dims.
TORCH_API bool has_leading_dims(at::OptionalIntArrayRef num_leading_dims);

} // namespace torch::distributed::fsdp
