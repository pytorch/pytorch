#pragma once

#include <ATen/core/Tensor.h>
#include <c10/macros/Export.h>
#include <c10/util/ArrayRef.h>
#include <utility>
#include <vector>

namespace c10d::fsdp::detail {

inline constexpr int64_t kCopyThreadsPerBlock = 128;
inline constexpr int64_t kChunkCatBytesPerBlock = kCopyThreadsPerBlock * 16;

TORCH_CUDA_CU_API std::pair<at::Tensor, std::vector<int64_t*>> pack_vecs(
    std::vector<const std::vector<int64_t>*> vecs,
    const at::Device& device);

TORCH_CUDA_CU_API void launch_split_with_sizes_copy(
    at::ArrayRef<int64_t*> ptrs,
    int64_t num_blocks,
    int64_t num_chunk_groups,
    int64_t src_stride,
    int64_t num_chunks);

TORCH_CUDA_CU_API void launch_chunk_cat(
    const at::Tensor& out,
    at::ArrayRef<int64_t*> ptrs,
    int64_t num_blocks_per_chunk,
    int64_t num_chunks,
    int64_t leading_dim,
    int64_t slice_size,
    int64_t chunk_size,
    at::ScalarType src_dtype);

} // namespace c10d::fsdp::detail
