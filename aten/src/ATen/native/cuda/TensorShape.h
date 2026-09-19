#pragma once

#include <ATen/core/Tensor.h>
#include <c10/macros/Export.h>
#include <vector>

namespace at::native::detail {

inline constexpr int64_t kChunkCatBytesPerBlock = 128 * 16;

struct ChunkCatMetadata {
  int64_t chunk_size;
  int64_t leading_dim;
  int64_t num_blocks_per_chunk;
  int64_t slice_size;
  std::vector<int64_t> srcs;
  std::vector<int64_t> block_idx_to_tensor_idx;
  std::vector<int64_t> tensor_idx_to_start_tensor_bytes;
  std::vector<int64_t> start_block_idx_per_tensor_chunk;
  std::vector<int64_t> actual_tensor_sizes;
  std::vector<int64_t> pad_tensor_chunk_sizes;
  std::vector<int64_t> num_blocks_per_tensor_chunk;
};

TORCH_CUDA_CU_API void launch_split_with_sizes_copy(
    const Device& device,
    const std::vector<int64_t>& srcs,
    const std::vector<int64_t>& dsts,
    const std::vector<int64_t>& chunk_sizes,
    int64_t src_stride,
    int64_t num_chunks);

TORCH_CUDA_CU_API void launch_chunk_cat(
    const Tensor& out,
    const ChunkCatMetadata& metadata,
    int64_t num_chunks,
    ScalarType src_dtype);

} // namespace at::native::detail
