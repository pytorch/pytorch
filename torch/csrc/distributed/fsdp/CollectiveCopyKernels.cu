#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/cuda/CUDAContext.h>
#include <ATen/ops/empty.h>
#include <c10/util/TypeCast.h>
#include <torch/csrc/distributed/fsdp/CollectiveCopyCUDA.hpp>
#include <algorithm>
#include <cstring>
#include <type_traits>

namespace c10d::fsdp::detail {

using c10::static_cast_with_inter_type;

// Mirrors the copy kernels in aten/src/ATen/native/cuda/TensorShape.cu.
// Raw descriptors preserve those algorithms without Tensor views for each outer slice.
static constexpr int64_t BLOCK_SIZE = 128;
static constexpr int64_t BYTES_PER_THREAD = 16;
static constexpr int64_t BYTES_PER_BLOCK = BYTES_PER_THREAD * BLOCK_SIZE;
static_assert(BLOCK_SIZE == kCopyThreadsPerBlock);
static_assert(BYTES_PER_BLOCK == kChunkCatBytesPerBlock);

static __host__ __device__ inline int64_t div_up(int64_t a, int64_t b) {
  return (a + b - 1) / b;
}

template <typename T>
__device__ inline void stream_load128(uint4& val, const T* addr) {
  uint64_t low, high;
#if defined(USE_ROCM) || (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 800))
  low = reinterpret_cast<const uint64_t*>(addr)[0];
  high = reinterpret_cast<const uint64_t*>(addr)[1];
#else
  asm("ld.global.nc.v2.u64 {%0, %1}, [%2];"
      : "=l"(low), "=l"(high)
      : "l"(addr));
#endif
  reinterpret_cast<uint64_t*>(&val)[0] = low;
  reinterpret_cast<uint64_t*>(&val)[1] = high;
}

template <typename T>
__device__ inline void stream_store128(T* addr, const uint4& val) {
  uint64_t low, high;
  low = reinterpret_cast<const uint64_t*>(&val)[0];
  high = reinterpret_cast<const uint64_t*>(&val)[1];
#if defined(USE_ROCM) || (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 800))
  reinterpret_cast<uint64_t*>(addr)[0] = low;
  reinterpret_cast<uint64_t*>(addr)[1] = high;
#else
  asm("st.global.cs.v2.u64 [%0], {%1, %2};" : : "l"(addr), "l"(low), "l"(high));
#endif
}

template <typename T>
static __device__ inline bool is_aligned(const void* addr) {
  return reinterpret_cast<uintptr_t>(addr) % sizeof(T) == 0;
}

template <typename T>
static __device__ inline void load128(uint4& val, const char* addr) {
  for (size_t i = 0; i < detail::BYTES_PER_THREAD / sizeof(T); ++i) {
    reinterpret_cast<T*>(&val)[i] = reinterpret_cast<const T*>(addr)[i];
  }
}

template <>
__device__ inline void load128<uint4>(uint4& val, const char* addr) {
  stream_load128(val, addr);
}

static __device__ inline void load128(uint4& val, const char* addr) {
  if (is_aligned<uint4>(addr)) {
    load128<uint4>(val, addr);
  } else if (is_aligned<int64_t>(addr)) {
    load128<uint64_t>(val, addr);
  } else if (is_aligned<uint32_t>(addr)) {
    load128<uint32_t>(val, addr);
  } else {
    load128<uint8_t>(val, addr);
  }
}

static __device__ __inline__ void get_aligned_region(
    char* ptr,
    const int64_t chunk_size,
    const int64_t alignment,
    int64_t& align_off,
    int64_t& aligned_size) {
  const int64_t ptr_val = reinterpret_cast<uintptr_t>(ptr);
  align_off = detail::div_up(ptr_val, alignment) * alignment - ptr_val;
  aligned_size = (chunk_size - align_off) / alignment * alignment;
}

static __device__ __inline__ void copy_chunk(
    char* dst,
    const char* src,
    int64_t chunk_size,
    int64_t thread_idx,
    int64_t num_threads) {
  if (chunk_size < num_threads) {
    if (thread_idx < chunk_size) {
      dst[thread_idx] = src[thread_idx];
    }
    return;
  }

  // Identify the region in which writes are guaranteed to be 128-bit aligned
  int64_t align_off, aligned_size;
  get_aligned_region(
      dst, chunk_size, detail::BYTES_PER_THREAD, align_off, aligned_size);

  for (int64_t off = align_off + thread_idx * detail::BYTES_PER_THREAD;
       off < align_off + aligned_size;
       off += num_threads * detail::BYTES_PER_THREAD) {
    uint4 val;
    // Oppurtunistically vectorize reads
    load128(val, &src[off]);
    stream_store128(&dst[off], val);
  }

  // Handle unaligned regions
  if (thread_idx < align_off && thread_idx < chunk_size) {
    dst[thread_idx] = src[thread_idx];
  }
  if (align_off + aligned_size + thread_idx < chunk_size) {
    dst[align_off + aligned_size + thread_idx] =
        src[align_off + aligned_size + thread_idx];
  }
}

static __global__ void split_with_sizes_copy_out_contiguous_no_cast_kernel(
    char** dst_base_addrs,
    char** src_base_addrs,
    int64_t* split_chunk_sizes,
    int64_t* block_idx_to_split_idx,
    int64_t* blocks_cumsums,
    int64_t src_stride,
    int64_t num_chunks) {
  const int64_t split_idx = block_idx_to_split_idx[blockIdx.x];
  const int64_t split_blocks =
      blocks_cumsums[split_idx + 1] - blocks_cumsums[split_idx];
  const int64_t split_threads = split_blocks * blockDim.x;
  const int64_t split_thread_idx =
      (blockIdx.x - blocks_cumsums[split_idx]) * blockDim.x + threadIdx.x;
  const int64_t split_chunk_size = split_chunk_sizes[split_idx];

  char* dst_base_addr = dst_base_addrs[split_idx];
  char* src_base_addr = src_base_addrs[split_idx];

  for (int64_t i = blockIdx.y; i < num_chunks; i += gridDim.y) {
    copy_chunk(
        dst_base_addr + i * split_chunk_size,
        src_base_addr + i * src_stride,
        split_chunk_size,
        split_thread_idx,
        split_threads);
  }
}

std::pair<at::Tensor, std::vector<int64_t*>> pack_vecs(
    std::vector<const std::vector<int64_t>*> vecs,
    const at::Device& device) {
  int64_t numel = 0;
  for (const auto* vec : vecs) {
    numel += vec->size();
  }

  auto packed = at::empty(
      {numel}, at::TensorOptions().dtype(at::kLong).pinned_memory(true));
  size_t offset = 0;
  for (const auto* vec : vecs) {
    memcpy(
        packed.data_ptr<int64_t>() + offset,
        vec->data(),
        sizeof(int64_t) * vec->size());
    offset += vec->size();
  }
  packed = packed.to(device, /*non_blocking=*/true);

  std::vector<int64_t*> ptrs;
  ptrs.reserve(vecs.size());
  offset = 0;
  for (const auto* vec : vecs) {
    ptrs.push_back(packed.data_ptr<int64_t>() + offset);
    offset += vec->size();
  }
  return std::make_pair(std::move(packed), std::move(ptrs));
}

template <typename dst_t, typename src_t>
__device__ __inline__ void copy_chunk_with_pad(
    dst_t* dst_ptr,
    src_t* src_ptr,
    int64_t max_chunk_size,
    int64_t actual_chunk_size,
    int64_t thread_idx,
    int64_t num_threads) {
  // Supports type cast
  if (!std::is_same_v<dst_t, src_t>) {
    const int64_t max_num_elems = max_chunk_size / sizeof(dst_t);
    const int64_t actual_num_elems = actual_chunk_size / sizeof(src_t);
    int64_t elem_index = thread_idx;
    while (elem_index < actual_num_elems) {
      dst_ptr[elem_index] =
          static_cast_with_inter_type<dst_t, src_t>::apply(src_ptr[elem_index]);
      elem_index += num_threads;
    }
    while (elem_index < max_num_elems) {
      dst_ptr[elem_index] = static_cast_with_inter_type<dst_t, int>::apply(0);
      elem_index += num_threads;
    }
    return;
  }
  char* dst = reinterpret_cast<char*>(dst_ptr);
  char* src = reinterpret_cast<char*>(src_ptr);
  // Fast path when the number of threads is larger than the number of bytes to
  // be copied (i.e., max_chunk_size). In this case, each thread only copies 1
  // byte. For 0 <= thread_idx < actual_chunk_size, the thread copies data from
  // `src`. For actual_chunk_size <= thread_idx < max_chunk_size, the thread set
  // the val=0 for padding.
  if (max_chunk_size < num_threads) {
    char val = static_cast<char>(0);
    if (thread_idx < actual_chunk_size) {
      val = src[thread_idx];
    }
    if (thread_idx < max_chunk_size) {
      dst[thread_idx] = val;
    }
    return;
  }
  // Split dst array into three parts:
  // [dst, dst+align_off), [dst+align_off, dst+align_end), [dst+align_end,
  // dst+max_chunk_size) The second part is aligned with BYTES_PER_THREAD(=16
  // bytes) to enable `stream_store128`.
  int64_t align_off, aligned_size;
  get_aligned_region(
      dst, actual_chunk_size, BYTES_PER_THREAD, align_off, aligned_size);
  int64_t align_end = align_off + aligned_size;
  for (int64_t i = align_off + thread_idx * BYTES_PER_THREAD; i < align_end;
       i += num_threads * BYTES_PER_THREAD) {
    uint4 val;
    if (is_aligned<uint4>(src + i)) {
      stream_load128(val, src + i);
    } else {
      for (size_t j = 0; j < BYTES_PER_THREAD; ++j) {
        reinterpret_cast<char*>(&val)[j] = src[i + j];
      }
    }
    stream_store128(&dst[i], val);
  }
  // Copy data for the first part of dst array [dst, dst+align_off).
  // Check `thread_idx<max_chunk_sze` for the edge case that max_chunk_size <
  // align_off.
  if (thread_idx < align_off && thread_idx < max_chunk_size) {
    char val = (char)0;
    if (thread_idx < actual_chunk_size) {
      val = src[thread_idx];
    }
    dst[thread_idx] = val;
  }
  // Copy data for the third part of dst array [dst+align_end,
  // dst+max_chunk_size).
  while (align_end + thread_idx < max_chunk_size) {
    char val = (char)0;
    if (align_end + thread_idx < actual_chunk_size) {
      val = src[align_end + thread_idx];
    }
    dst[align_end + thread_idx] = val;
    align_end += num_threads;
  }
}

// NOTE [CUDA kernel for chunk_cat]
// chunk_cat_cuda adopts a "jagged grid" strategy, inspired by NOTE [CUDA fast
// path for split_with_sizes_copy.out]. In addition, chunk_cat_cuda supports
// padding via copy_chunk_with_pad when src chunk size is less than dst chunk
// size.
template <typename dst_t, typename src_t>
static __global__ void chunk_cat_cuda_kernel(
    src_t** src,
    dst_t* dst,
    int64_t* block_idx_to_tensor_idx,
    int64_t* tensor_idx_to_start_tensor_bytes,
    int64_t* start_block_idx_per_tensor_chunk,
    int64_t* actual_tensor_sizes,
    int64_t* pad_tensor_chunk_sizes,
    int64_t* num_blocks_per_tensor_chunk,
    int64_t slice_size,
    int64_t chunk_size,
    int64_t dst_to_src_ratio) {
  const int64_t slice_idx = blockIdx.z;
  const int64_t chunk_idx = blockIdx.y;
  const int64_t tensor_idx = block_idx_to_tensor_idx[blockIdx.x];
  const int64_t tile_idx =
      blockIdx.x - start_block_idx_per_tensor_chunk[tensor_idx];
  // Number of threads for the `tensor_idx`-th tensor chunk.
  const int64_t num_threads =
      num_blocks_per_tensor_chunk[tensor_idx] * BLOCK_SIZE;
  const int64_t thread_idx = tile_idx * BLOCK_SIZE + threadIdx.x;
  char* src_addr = reinterpret_cast<char**>(src)[tensor_idx] +
      slice_idx * actual_tensor_sizes[tensor_idx] +
      chunk_idx * pad_tensor_chunk_sizes[tensor_idx] / dst_to_src_ratio;
  char* dst_addr = reinterpret_cast<char*>(dst) + slice_idx * slice_size +
      chunk_idx * chunk_size + tensor_idx_to_start_tensor_bytes[tensor_idx];
  // Compute the actual number of bytes to copy from src.
  const int64_t actual_copy_size = std::min(
      pad_tensor_chunk_sizes[tensor_idx] / dst_to_src_ratio,
      std::max(
          (int64_t)0,
          actual_tensor_sizes[tensor_idx] -
              chunk_idx * pad_tensor_chunk_sizes[tensor_idx] /
                  dst_to_src_ratio));
  copy_chunk_with_pad<dst_t, src_t>(
      reinterpret_cast<dst_t*>(dst_addr),
      reinterpret_cast<src_t*>(src_addr),
      pad_tensor_chunk_sizes[tensor_idx],
      actual_copy_size,
      thread_idx,
      num_threads);
}

void launch_split_with_sizes_copy(
    at::ArrayRef<int64_t*> ptrs,
    int64_t num_blocks,
    int64_t num_chunk_groups,
    int64_t src_stride,
    int64_t num_chunks) {
  dim3 blocks(num_blocks, num_chunk_groups, 1);
  dim3 threads(BLOCK_SIZE, 1, 1);
  split_with_sizes_copy_out_contiguous_no_cast_kernel<<<
      blocks,
      threads,
      0,
      at::cuda::getCurrentCUDAStream()>>>(
      /*dst_base_addrs=*/reinterpret_cast<char**>(ptrs[0]),
      /*src_base_addrs=*/reinterpret_cast<char**>(ptrs[1]),
      /*split_chunk_sizes=*/ptrs[2],
      /*block_idx_to_split_idx=*/ptrs[3],
      /*blocks_cumsums=*/ptrs[4],
      src_stride,
      num_chunks);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename dst_t, typename src_t>
static void launch_chunk_cat_typed(
    const at::Tensor& out,
    at::ArrayRef<int64_t*> ptrs,
    dim3 blocks,
    int64_t slice_size,
    int64_t chunk_size) {
  dim3 threads(BLOCK_SIZE, 1, 1);
  chunk_cat_cuda_kernel<<<
      blocks,
      threads,
      0,
      at::cuda::getCurrentCUDAStream()>>>(
      /*srcs=*/reinterpret_cast<src_t**>(ptrs[0]),
      reinterpret_cast<dst_t*>(out.data_ptr()),
      /*block_idx_to_tensor_idx=*/ptrs[1],
      /*tensor_idx_to_start_tensor_bytes=*/ptrs[2],
      /*start_block_idx_per_tensor_chunk=*/ptrs[3],
      /*actual_tensor_sizes=*/ptrs[4],
      /*pad_tensor_chunk_sizes=*/ptrs[5],
      /*num_blocks_per_tensor_chunk=*/ptrs[6],
      slice_size,
      chunk_size,
      sizeof(dst_t) / sizeof(src_t));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_chunk_cat(
    const at::Tensor& out,
    at::ArrayRef<int64_t*> ptrs,
    int64_t num_blocks_per_chunk,
    int64_t num_chunks,
    int64_t leading_dim,
    int64_t slice_size,
    int64_t chunk_size,
    at::ScalarType src_dtype) {
  dim3 blocks(num_blocks_per_chunk, num_chunks, leading_dim);
  if (src_dtype == at::kBFloat16 && out.scalar_type() == at::kFloat) {
    launch_chunk_cat_typed<float, at::BFloat16>(
        out, ptrs, blocks, slice_size, chunk_size);
  } else {
    TORCH_INTERNAL_ASSERT(src_dtype == out.scalar_type());
    launch_chunk_cat_typed<char, char>(out, ptrs, blocks, slice_size, chunk_size);
  }
}

} // namespace c10d::fsdp::detail
