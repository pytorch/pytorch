#pragma once

#include <ATen/ATen.h>
#include <torch/csrc/Export.h>

namespace torch::distributed::fsdp {

// Composite kernel of fsdp::chunk_cat_mixed_dtype: calls _chunk_cat.out along
// dim 0, first casting each input to out's dtype if the input dtypes differ.
TORCH_API void chunk_cat_mixed_dtype(
    at::TensorList tensors,
    int64_t num_chunks,
    at::Tensor& out);

} // namespace torch::distributed::fsdp
