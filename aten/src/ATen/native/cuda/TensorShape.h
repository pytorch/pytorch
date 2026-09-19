#pragma once

#include <ATen/core/Tensor.h>
#include <c10/macros/Export.h>
#include <c10/util/ArrayRef.h>
#include <utility>
#include <vector>

namespace at::native::detail {

inline constexpr int64_t kCopyThreadsPerBlock = 128;
inline constexpr int64_t kChunkCatBytesPerBlock = kCopyThreadsPerBlock * 16;

TORCH_CUDA_CU_API std::pair<Tensor, std::vector<int64_t*>> pack_vecs(
    std::vector<const std::vector<int64_t>*> vecs,
    const Device& device);

TORCH_CUDA_CU_API void launch_split_with_sizes_copy(
    ArrayRef<int64_t*> ptrs,
    int64_t num_blocks,
    int64_t num_chunk_groups,
    int64_t src_stride,
    int64_t num_chunks);

TORCH_CUDA_CU_API void launch_chunk_cat(
    const Tensor& out,
    ArrayRef<int64_t*> ptrs,
    int64_t num_blocks_per_chunk,
    int64_t num_chunks,
    int64_t leading_dim,
    int64_t slice_size,
    int64_t chunk_size,
    ScalarType src_dtype);

} // namespace at::native::detail
