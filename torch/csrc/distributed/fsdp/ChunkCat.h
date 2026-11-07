#pragma once

#include <ATen/ATen.h>
#include <torch/csrc/Export.h>

namespace torch::distributed::fsdp {

// Composite kernel of fsdp::chunk_cat_mixed_dtype. Without leading dims or
// padded dim sizes, it calls _chunk_cat.out, first casting each input to out's
// dtype if the input dtypes differ. With them, it copies each input into its
// slice of out, which casts as it copies, and zero-fills the padding of chunks
// whose rank gets fewer rows than padded_dim_sizes[i] / num_chunks (by default
// ceil(size / num_chunks)) of the chunked dim.
TORCH_API void chunk_cat_mixed_dtype(
    at::TensorList tensors,
    int64_t dim,
    int64_t num_chunks,
    at::Tensor& out,
    at::OptionalIntArrayRef num_leading_dims,
    at::OptionalIntArrayRef padded_dim_sizes = std::nullopt);

// Validates fsdp::chunk_cat_mixed_dtype inputs with leading dims or padded dim
// sizes, which fused backend kernels call themselves.
TORCH_API void check_chunk_cat_leading_dims_inputs(
    at::TensorList tensors,
    at::IntArrayRef num_leading_dims,
    int64_t num_chunks,
    const at::Tensor& out,
    at::OptionalIntArrayRef padded_dim_sizes = std::nullopt);

// Whether any input of fsdp::chunk_cat_mixed_dtype has leading dims.
TORCH_API bool has_leading_dims(at::OptionalIntArrayRef num_leading_dims);

// num_leading_dims, or a zero for every input if it is absent or empty.
TORCH_API std::vector<int64_t> leading_dims_or_zeros(
    at::OptionalIntArrayRef num_leading_dims,
    size_t num_tensors);

} // namespace torch::distributed::fsdp
