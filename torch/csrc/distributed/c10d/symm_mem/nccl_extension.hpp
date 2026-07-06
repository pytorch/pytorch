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

// In-place reshard of a 1-D, 2-D, or 3-D tensor between two rank meshes,
// backed by `ncclReshardWithWindow` from NCCL M2N or an NCCL build exporting
// the same API. `buf` must be allocated through NCCL symmetric memory and
// sized to hold the larger of the source and destination local shapes. Each
// mesh maps onto `ncclMesh_t::{dims, startRank, placement}`, where placement
// entries are -1 for REPLICATE or a non-negative tensor dim index for SHARD.
// A rank that does not own a tile on a given side passes a zero-shape on that
// side; the binding maps that to `ncclDistTensor_t::dataPtr = NULL`.
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

// Initialize NCCL M2N process-global state. `max_cta`, when set, maps to
// ncclM2nConfig_t::maxCta. Environment variables still take precedence in the
// NCCL M2N library. Call this before the first reshard to apply config.
TORCH_API void nccl_m2n_init(std::optional<int64_t> max_cta);

// Best-effort `ncclM2nFinalize`; releases the reshard library's
// internal caches and transpose buffer. Idempotent and safe to call
// multiple times before tearing down CUDA and NCCL contexts.
TORCH_API void nccl_m2n_finalize();
} // namespace c10d::nccl_extension
