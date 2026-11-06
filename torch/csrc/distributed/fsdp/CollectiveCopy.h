#pragma once

#include <ATen/core/Tensor.h>

namespace torch::distributed::fsdp {

// Composite kernel of fsdp::_all_gather_copy_out_. It validates its inputs with
// the check function, which fused backend kernels call themselves.
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

} // namespace torch::distributed::fsdp
