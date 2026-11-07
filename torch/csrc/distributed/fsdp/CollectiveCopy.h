#pragma once

#include <ATen/core/Tensor.h>

namespace torch::distributed::fsdp {

// Composite kernel of fsdp::_all_gather_copy_out_. It validates its inputs with
// the check function, which fused backend kernels call themselves.
//
// Without output_inner_sizes, out[i] holds every rank's chunk: viewed as
// (outer_sizes[i], -1), each row concatenates the num_chunks chunks of
// split_sizes[i] / outer_sizes[i] elements. With it, each such row holds only
// its first output_inner_sizes[i] elements, which may end inside a rank's
// chunk or before later ranks' chunks (their rows are padding and are not
// copied). An out[i] with outer_sizes[i] > 1 then has exactly
// outer_sizes[i] * output_inner_sizes[i] elements; one with outer_sizes[i] == 1
// may be larger, and only its prefix is written.
TORCH_API void check_all_gather_copy_out_inputs(
    at::TensorList out,
    const at::Tensor& input,
    c10::SymIntArrayRef split_sizes,
    c10::SymIntArrayRef outer_sizes,
    int64_t num_chunks,
    at::OptionalSymIntArrayRef output_inner_sizes);

TORCH_API void all_gather_copy_out(
    at::TensorList out,
    const at::Tensor& input,
    c10::SymIntArrayRef split_sizes,
    c10::SymIntArrayRef outer_sizes,
    int64_t num_chunks,
    at::OptionalSymIntArrayRef output_inner_sizes);

} // namespace torch::distributed::fsdp
