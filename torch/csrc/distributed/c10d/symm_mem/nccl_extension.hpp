#pragma once

#include <ATen/ATen.h>
#include <c10/macros/Macros.h>

namespace c10d::nccl_extension {

TORCH_API bool is_nccl_symmem_available();

TORCH_API void nccl_put(at::Tensor& tensor, const int64_t peer);

TORCH_API void nccl_get(at::Tensor& tensor, const int64_t peer);

TORCH_API at::Tensor& nccl_get_out(
    at::Tensor& dst,
    const at::Tensor& src,
    int64_t peer,
    const std::string& group_name);

TORCH_API void nccl_wait_for_signal(at::Tensor& sigpad, int64_t signal);

TORCH_API void nccl_put_with_signal(
    at::Tensor& tensor,
    int64_t signal,
    int64_t peer);

// Simultaneously reduce N blocks of a 2-D input tensor from a shared symmetric
// memory buffer, routing each to a specific destination rank. Blocks are
// described by inclusive-prefix-sum offsets along `dim` (0 or 1); all blocks
// must have equal size.
TORCH_API void nccl_reduce_scatter_offset(
    const at::Tensor& input,
    at::TensorList out,
    const std::string& group_name,
    int64_t dim,
    std::optional<at::IntArrayRef> offsets,
    std::optional<at::IntArrayRef> dst_ranks,
    const std::string& red_op);
} // namespace c10d::nccl_extension
