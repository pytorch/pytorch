#pragma once

#include <ATen/ATen.h>
#include <c10/macros/Macros.h>
#include <torch/csrc/distributed/c10d/symm_mem/SymmetricMemory.hpp>

#include <optional>

namespace c10d::nccl_extension {

TORCH_API bool is_nccl_symmem_available();

TORCH_API void nccl_put(at::Tensor& tensor, const int64_t peer);

TORCH_API void nccl_get(at::Tensor& tensor, const int64_t peer);

TORCH_API void nccl_get_out(
    at::Tensor& dst,
    const c10::intrusive_ptr<c10d::symmetric_memory::SymmetricMemory>& hdl,
    int64_t offset,
    int64_t size,
    int64_t peer);

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

// Reshard a 1-D, 2-D, or 3-D tensor with `ncclReshardWithWindow`. `buf` must
// use NCCL symmetric memory and hold the larger local shape. Meshes use
// `ncclMesh_t::{dims, startRank}`; placements use
// `ncclDistTensor_t::placements`. Every rank passes the same shape metadata;
// `dataPtr = NULL` marks a rank without data on that side.
TORCH_API void nccl_reshard(
    at::Tensor& buf,
    at::IntArrayRef src_local_shape,
    at::IntArrayRef src_mesh_dims,
    int64_t src_mesh_start_rank,
    at::IntArrayRef src_placement,
    at::IntArrayRef dst_local_shape,
    at::IntArrayRef dst_mesh_dims,
    int64_t dst_mesh_start_rank,
    at::IntArrayRef dst_placement,
    const std::string& group_name);

// Initialize NCCL M2N state. Call before reshard to set `maxCta`.
TORCH_API void nccl_m2n_init(std::optional<int64_t> max_cta);

// Return whether this build includes the NCCL M2N API.
TORCH_API bool nccl_m2n_is_available();

// Release NCCL M2N state before tearing down CUDA and NCCL contexts.
TORCH_API void nccl_m2n_finalize();
} // namespace c10d::nccl_extension
