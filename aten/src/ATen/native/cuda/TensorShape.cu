#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/Dispatch.h>
#include <ATen/core/Tensor.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/native/Resize.h>
#include <ATen/native/TensorShape.h>
#include <c10/cuda/CUDAGraphsC10Utils.h>
#include <c10/util/TypeCast.h>
#include <ATen/native/cuda/MemoryAccess.cuh>

#include <algorithm>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_chunk_cat_native.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/split_with_sizes_copy_native.h>
#endif

namespace at::native {

namespace detail {

// NOTE [CUDA fast path for split_with_sizes_copy.out]
// split_with_sizes_copy.out for contiguous operands has the following
// properties:
// - Each src split consists of multiple chunks that are separated by a fixed
// stride. The number of chunks and the strides are the same across all src
// splits.
// - Each dst split is the concatenation of the chunks in its corresponding src
// splits.
// - The sizes of chunks vary across splits.
// - A (src, dst) chunk pair is not guaranteed to have the
// same alignment.
//
// The following strategies are employed to optimize for this workload:
// - The entire workload is fused into a single kernel to maximize I/O
// throughput and minimize wave quantization.
// - To account for both small and large chunk sizes, a "jagged grid" is used.
// Each chunk is processed by one or more blocks depending on its size.
// - Within each chunk, the region in which writes can be vectorized is
// identified. Within this region, writes are always vectorized and reads are
// oppurtunistically vectorized.
static constexpr int64_t BLOCK_SIZE = 128;
static constexpr int64_t BYTES_PER_THREAD = 16;
static constexpr int64_t BYTES_PER_BLOCK = BYTES_PER_THREAD * BLOCK_SIZE;
// 16-byte dst vectors each thread converts per step in the vectorized cast.
static constexpr int64_t CAST_VECS_PER_THREAD = 2;
// The same, on SM 10.7 for calls above a tuned size, see
// get_cast_vecs_per_thread.
static constexpr int64_t SM107_WIDE_CAST_VECS_PER_THREAD = 4;
static_assert(
    SM107_WIDE_CAST_VECS_PER_THREAD != CAST_VECS_PER_THREAD,
    "kernels with SM107_WIDE_CAST_VECS_PER_THREAD compile to a stub outside "
    "SM 10.x, so the default must use a different value");

static __host__ __device__ inline int64_t div_up(int64_t a, int64_t b) {
  return (a + b - 1) / b;
}

// Whether the mixed-dtype copy (bf16 -> fp32 in FSDP) vectorizes a chunk of
// `chunk_bytes` dst bytes. The host sizes the grid with the same test, so a
// vectorized chunk gets cast_vecs_per_thread times fewer blocks.
static __host__ __device__ inline bool is_vectorized_cast(int64_t chunk_bytes) {
  // Tuned: smaller chunks are faster with the per-element loop.
  constexpr int64_t kMinVectorizedCastBytes = 8 * 1024;
  return chunk_bytes >= kMinVectorizedCastBytes;
}

// Vectors per thread per step for the vectorized chunks of a mixed-dtype call
// that writes `out_bytes` on `device`.
static int64_t get_cast_vecs_per_thread(
    int64_t out_bytes,
    c10::DeviceIndex device) {
  // Tuned on SM 10.7 with SM107_WIDE_CAST_VECS_PER_THREAD = 4 and
  // CAST_VECS_PER_THREAD = 2: the wide one was faster at and above this size,
  // slower below.
  constexpr int64_t kMinSm107WideCastBytes = 8 * 1024 * 1024;
  const cudaDeviceProp* prop = at::cuda::getDeviceProperties(device);
  const bool sm107 = prop->major == 10 && prop->minor == 7;
  return sm107 && out_bytes >= kMinSm107WideCastBytes
      ? SM107_WIDE_CAST_VECS_PER_THREAD
      : CAST_VECS_PER_THREAD;
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

// Calculate the base addr for each split.
static inline std::vector<int64_t> get_split_base_addrs(
    const at::Tensor& tensor,
    at::IntArrayRef split_sizes,
    int64_t dim) {
  const auto* data_ptr = static_cast<const char*>(tensor.const_data_ptr());
  const auto strides = tensor.strides();
  const auto element_sz = tensor.element_size();
  int64_t off = 0;
  std::vector<int64_t> split_base_addrs;
  split_base_addrs.reserve(split_sizes.size());
  for (const auto& split_size : split_sizes) {
    split_base_addrs.push_back(reinterpret_cast<int64_t>(data_ptr + off));
    off += split_size * strides[dim] * element_sz;
  }
  return split_base_addrs;
}

static inline std::vector<int64_t> get_dst_addrs(at::TensorList out) {
  std::vector<int64_t> addrs;
  addrs.reserve(out.size());
  for (const auto& tensor : out) {
    addrs.push_back(reinterpret_cast<int64_t>(tensor.data_ptr()));
  }
  return addrs;
}

// Calculate the chunk size for each split in bytes.
static inline std::vector<int64_t> get_split_chunk_sizes(
    const at::Tensor& tensor,
    at::IntArrayRef split_sizes,
    int64_t dim) {
  const auto stride = tensor.stride(dim);
  const auto element_sz = tensor.element_size();
  std::vector<int64_t> split_chunk_sizes;
  split_chunk_sizes.reserve(split_sizes.size());
  for (const auto& split_size : split_sizes) {
    split_chunk_sizes.push_back(split_size * stride * element_sz);
  }
  return split_chunk_sizes;
}

// Calculate the chunk stride in bytes. This is the same for all splits.
static inline int64_t get_chunk_stride(const at::Tensor& tensor, int64_t dim) {
  int64_t stride = 1;
  for (int64_t d = dim; d < tensor.dim(); ++d) {
    stride *= tensor.sizes()[d];
  }
  return stride * tensor.element_size();
}

// Calculate the number of chunks. This is the same for all splits.
static inline int64_t get_num_chunks(const at::Tensor& tensor, int64_t dim) {
  int64_t num_chunks = tensor.numel();
  for (int64_t d = dim; d < tensor.dim(); ++d) {
    num_chunks /= tensor.sizes()[d];
  }
  return num_chunks;
}

// Pack multiple std::vector<int64_t> into a single cuda tensor.
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

static inline std::vector<int64_t> get_chunk_cat_out_sizes(
    IntArrayRef input_tensor_sizes,
    int64_t dim,
    int64_t num_chunks,
    int64_t chunk_size,
    int64_t out_element_size) {
  std::vector<int64_t> view_sizes = std::vector<int64_t>(
      input_tensor_sizes.begin(), input_tensor_sizes.begin() + dim);
  view_sizes.insert(
      view_sizes.end(), {num_chunks, chunk_size / out_element_size});
  return view_sizes;
}

// Returns dst element i of a chunk: src element i cast to dst_t, or zero
// padding past the `actual_num_elems` src elements (src is not read there).
template <typename dst_t, typename src_t>
static __device__ inline dst_t cast_or_pad(
    const src_t* src_ptr,
    int64_t actual_num_elems,
    int64_t i) {
  return i < actual_num_elems
      ? static_cast_with_inter_type<dst_t, src_t>::apply(src_ptr[i])
      : static_cast_with_inter_type<dst_t, int>::apply(0);
}

// Returns dst elements [first_elem, first_elem + vec_size) as one 16-byte
// vector. Reads src with one vector load when that range has no padding and
// src is aligned for it, else element by element.
template <typename dst_t, typename src_t>
static __device__ inline uint4 cast_or_pad_vec(
    const src_t* src_ptr,
    int64_t first_elem,
    int64_t actual_num_elems) {
  constexpr int64_t vec_size = BYTES_PER_THREAD / sizeof(dst_t);
  using src_vec_t = memory::aligned_vector<src_t, vec_size>;
  uint4 dst_vec;
  dst_t* dst_elems = reinterpret_cast<dst_t*>(&dst_vec);
  const src_t* src_vec_ptr = src_ptr + first_elem;
  if (first_elem + vec_size <= actual_num_elems &&
      is_aligned<src_vec_t>(src_vec_ptr)) {
    const src_vec_t src_vec = *reinterpret_cast<const src_vec_t*>(src_vec_ptr);
#pragma unroll
    for (int64_t j = 0; j < vec_size; ++j) {
      dst_elems[j] =
          static_cast_with_inter_type<dst_t, src_t>::apply(src_vec.val[j]);
    }
  } else {
#pragma unroll
    for (int64_t j = 0; j < vec_size; ++j) {
      dst_elems[j] =
          cast_or_pad<dst_t>(src_ptr, actual_num_elems, first_elem + j);
    }
  }
  return dst_vec;
}

// Vectorized version of the per-element cast in copy_chunk_with_pad. The dst
// chunk is split into
//   head [0, body_begin): elements before the first 16-byte aligned address,
//   body [body_begin, body_end): 16-byte vectors, stored with stream_store128,
//   tail [body_end, max_num_elems): the rest, less than one vector.
// Each thread converts `vecs_per_thread` vectors per step and issues all their
// loads before any store, to keep more bytes in flight. Only called for chunks
// is_vectorized_cast accepts, so head and tail lie inside the chunk.
template <int64_t vecs_per_thread, typename dst_t, typename src_t>
static __device__ inline void cast_chunk_with_pad_vectorized(
    dst_t* dst_ptr,
    const src_t* src_ptr,
    int64_t max_num_elems,
    int64_t actual_num_elems,
    int64_t thread_idx,
    int64_t num_threads) {
  constexpr int64_t vec_size = BYTES_PER_THREAD / sizeof(dst_t);
  int64_t head_bytes, body_bytes;
  get_aligned_region(
      reinterpret_cast<char*>(dst_ptr),
      max_num_elems * sizeof(dst_t),
      BYTES_PER_THREAD,
      head_bytes,
      body_bytes);
  const int64_t body_begin = head_bytes / sizeof(dst_t);
  const int64_t num_vecs = body_bytes / BYTES_PER_THREAD;
  const int64_t body_end = body_begin + num_vecs * vec_size;
  for (int64_t first_vec = thread_idx; first_vec < num_vecs;
       first_vec += num_threads * vecs_per_thread) {
    uint4 vecs[vecs_per_thread];
#pragma unroll
    for (int64_t u = 0; u < vecs_per_thread; ++u) {
      const int64_t v = first_vec + u * num_threads;
      if (v < num_vecs) {
        vecs[u] = cast_or_pad_vec<dst_t>(
            src_ptr, body_begin + v * vec_size, actual_num_elems);
      }
    }
#pragma unroll
    for (int64_t u = 0; u < vecs_per_thread; ++u) {
      const int64_t v = first_vec + u * num_threads;
      if (v < num_vecs) {
        stream_store128(dst_ptr + body_begin + v * vec_size, vecs[u]);
      }
    }
  }
  // Head and tail are shorter than a vector, so one pass of threads suffices.
  if (thread_idx < body_begin) {
    dst_ptr[thread_idx] =
        cast_or_pad<dst_t>(src_ptr, actual_num_elems, thread_idx);
  }
  const int64_t tail_idx = body_end + thread_idx;
  if (tail_idx < max_num_elems) {
    dst_ptr[tail_idx] = cast_or_pad<dst_t>(src_ptr, actual_num_elems, tail_idx);
  }
}

// Copy `max_chunk_size` bytes from `src` to `dst` by `num_threads`, and pad
// zero when `src` size (i.e., actual_chunk_size) is less than `max_chunk_size`.
// src elements are cast to dst_t when the two types differ, with
// `cast_vecs_per_thread` vectors per thread per step (0: per element only).
// torch/csrc/distributed/fsdp/ChunkCat.cu has a copy of this function without
// the vectorized cast.
template <typename dst_t, typename src_t, int64_t cast_vecs_per_thread>
__device__ __inline__ void copy_chunk_with_pad(
    dst_t* dst_ptr,
    src_t* src_ptr,
    int64_t max_chunk_size,
    int64_t actual_chunk_size,
    int64_t thread_idx,
    int64_t num_threads) {
  // Supports type cast
  if constexpr (!std::is_same_v<dst_t, src_t>) {
    const int64_t max_num_elems = max_chunk_size / sizeof(dst_t);
    const int64_t actual_num_elems = actual_chunk_size / sizeof(src_t);
    if constexpr (cast_vecs_per_thread > 0) {
      if (is_vectorized_cast(max_chunk_size)) {
        cast_chunk_with_pad_vectorized<cast_vecs_per_thread>(
            dst_ptr,
            src_ptr,
            max_num_elems,
            actual_num_elems,
            thread_idx,
            num_threads);
        return;
      }
    }
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
template <typename dst_t, typename src_t, int64_t cast_vecs_per_thread>
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
  if constexpr (cast_vecs_per_thread == SM107_WIDE_CAST_VECS_PER_THREAD) {
#if __CUDA_ARCH__ / 100 != 10
    // Only launched on SM 10.7, so other archs compile a stub.
    CUDA_KERNEL_ASSERT(
        false &&
        "chunk_cat_cuda_kernel with SM107_WIDE_CAST_VECS_PER_THREAD "
        "supports only sm_10x");
    return;
#endif
  }
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
  copy_chunk_with_pad<dst_t, src_t, cast_vecs_per_thread>(
      reinterpret_cast<dst_t*>(dst_addr),
      reinterpret_cast<src_t*>(src_addr),
      pad_tensor_chunk_sizes[tensor_idx],
      actual_copy_size,
      thread_idx,
      num_threads);
}

bool all_contiguous(TensorList tensors) {
  for (const auto& t : tensors) {
    if (!t.is_contiguous()) {
      return false;
    }
  }
  return true;
}

// Get leading dimensions before `dim`-th dimension.
static inline int64_t get_leading_dim(at::IntArrayRef sizes, int64_t dim) {
  int64_t leading_dim = 1;
  if (dim > 0) {
    leading_dim = c10::multiply_integers(sizes.slice(0, dim));
  }
  return leading_dim;
}

// Get trailing dimensions after `dim`-th dimension and padded size along
// `dim`-th dimension.
static inline std::pair<int64_t, int64_t> get_pad_size(
    at::IntArrayRef sizes,
    int64_t dim,
    int64_t num_chunks) {
  int64_t trailing_numel = 1;
  if (sizes.size() > (uint64_t)dim + 1) {
    trailing_numel =
        c10::multiply_integers(sizes.slice(dim + 1, sizes.size() - dim - 1));
  }
  int64_t pad_size_along_dim =
      detail::div_up(sizes[dim], num_chunks) * num_chunks;
  return std::make_pair(pad_size_along_dim, trailing_numel);
}

// Get the padded chunk size.
static inline int64_t get_chunk_size(
    TensorList tensors,
    int64_t dim,
    int64_t num_chunks,
    int64_t elem_size) {
  auto num_tensors = tensors.size();
  int64_t chunk_size = 0;
  for (const auto i : c10::irange(num_tensors)) {
    auto [pad_size_along_dim, trailing_numel] =
        get_pad_size(tensors[i].sizes(), dim, num_chunks);
    const int64_t pad_tensor_chunk_size =
        pad_size_along_dim * trailing_numel * elem_size / num_chunks;
    chunk_size += pad_tensor_chunk_size;
  }
  return chunk_size;
}

// Get metadata for chunk_cat.
std::tuple<
    int64_t,
    int64_t,
    int64_t,
    int64_t,
    std::vector<int64_t>,
    std::vector<int64_t>,
    std::vector<int64_t>,
    std::vector<int64_t>,
    std::vector<int64_t>,
    std::vector<int64_t>,
    std::vector<int64_t>>
get_chunk_cat_metadata(
    TensorList tensors,
    int64_t dim,
    int64_t num_chunks,
    int64_t dst_elem_size,
    int64_t src_elem_size,
    int64_t cast_vecs_per_thread) {
  TORCH_CHECK(
      dst_elem_size % src_elem_size == 0,
      "get_chunk_cat_metadata error: only support dst_elem_size % src_elem_size == 0");
  auto num_tensors = tensors.size();
  int64_t leading_dim = get_leading_dim(tensors[0].sizes(), dim);
  std::vector<int64_t> pad_tensor_chunk_sizes;
  std::vector<int64_t> num_blocks_per_tensor_chunk;
  std::vector<int64_t> start_block_idx_per_tensor_chunk{0};
  std::vector<int64_t> actual_tensor_sizes;
  std::vector<int64_t> tensor_idx_to_start_tensor_bytes{0};
  std::vector<int64_t> srcs;
  pad_tensor_chunk_sizes.reserve(num_tensors);
  num_blocks_per_tensor_chunk.reserve(num_tensors);
  start_block_idx_per_tensor_chunk.reserve(num_tensors + 1);
  actual_tensor_sizes.reserve(num_tensors);
  tensor_idx_to_start_tensor_bytes.reserve(num_tensors + 1);
  srcs.reserve(num_tensors);
  // block_idx_to_tensor_idx cannot be reserved since the number of blocks is
  // data dependent
  std::vector<int64_t> block_idx_to_tensor_idx;
  // Inline computing `chunk_size` to avoid redundant computation
  int64_t chunk_size = 0;
  for (const auto i : c10::irange(num_tensors)) {
    const at::Tensor& tensor = tensors[i];
    srcs.push_back(reinterpret_cast<int64_t>(tensor.data_ptr()));
    auto sizes = tensor.sizes();
    auto [pad_size_along_dim, trailing_numel] =
        get_pad_size(sizes, dim, num_chunks);
    const int64_t pad_tensor_chunk_size =
        pad_size_along_dim * trailing_numel * dst_elem_size / num_chunks;
    pad_tensor_chunk_sizes.push_back(pad_tensor_chunk_size);
    chunk_size += pad_tensor_chunk_size;
    // Number of blocks required to process this tensor chunk. A vectorized
    // cast writes cast_vecs_per_thread vectors per thread per step, so it
    // needs that many times fewer blocks.
    const bool vectorized_cast = cast_vecs_per_thread > 0 &&
        detail::is_vectorized_cast(pad_tensor_chunk_size);
    const int64_t bytes_per_block = vectorized_cast
        ? detail::BYTES_PER_BLOCK * cast_vecs_per_thread
        : detail::BYTES_PER_BLOCK;
    const int64_t num_blocks =
        detail::div_up(pad_tensor_chunk_size, bytes_per_block);
    num_blocks_per_tensor_chunk.push_back(num_blocks);
    start_block_idx_per_tensor_chunk.push_back(
        start_block_idx_per_tensor_chunk.back() + num_blocks);
    block_idx_to_tensor_idx.insert(
        block_idx_to_tensor_idx.end(), num_blocks, i);
    tensor_idx_to_start_tensor_bytes.push_back(
        tensor_idx_to_start_tensor_bytes.back() + pad_tensor_chunk_size);
    actual_tensor_sizes.push_back(sizes[dim] * trailing_numel * src_elem_size);
  }
  const int64_t num_blocks_per_chunk = start_block_idx_per_tensor_chunk.back();
  const int64_t slice_size = num_chunks * chunk_size;
  return std::make_tuple(
      chunk_size,
      leading_dim,
      num_blocks_per_chunk,
      slice_size,
      std::move(srcs),
      std::move(block_idx_to_tensor_idx),
      std::move(tensor_idx_to_start_tensor_bytes),
      std::move(start_block_idx_per_tensor_chunk),
      std::move(actual_tensor_sizes),
      std::move(pad_tensor_chunk_sizes),
      std::move(num_blocks_per_tensor_chunk));
}

// See [CUDA kernel for chunk_cat_cuda]
template <typename dst_t, typename src_t>
void _chunk_cat_out_cuda_contiguous(
    TensorList tensors,
    int64_t dim,
    int64_t num_chunks,
    Tensor& out,
    int64_t dst_elem_size,
    int64_t src_elem_size) {
  const auto device = tensors[0].device();
  // The vectorized cast's vectors per thread set the grid, so they are chosen
  // before the metadata, from the size of `out`.
  int64_t cast_vecs_per_thread = 0;
  if constexpr (!std::is_same_v<dst_t, src_t>) {
    const int64_t out_bytes =
        get_chunk_size(tensors, dim, num_chunks, dst_elem_size) * num_chunks *
        get_leading_dim(tensors[0].sizes(), dim);
    cast_vecs_per_thread = get_cast_vecs_per_thread(out_bytes, device.index());
  }
  // `get_chunk_cat_metadata` must return vectors and `pack_vecs` cannot be
  // moved into `get_chunk_cat_metadata`. Otherwise `packed` would point to
  // vectors allocated inside `get_chunk_cat_metadata` which become out of local
  // scope.
  auto
      [chunk_size,
       leading_dim,
       num_blocks_per_chunk,
       slice_size,
       srcs,
       block_idx_to_tensor_idx,
       tensor_idx_to_start_tensor_bytes,
       start_block_idx_per_tensor_chunk,
       actual_tensor_sizes,
       pad_tensor_chunk_sizes,
       num_blocks_per_tensor_chunk] =
          get_chunk_cat_metadata(
              tensors,
              dim,
              num_chunks,
              dst_elem_size,
              src_elem_size,
              cast_vecs_per_thread);
  auto packed = pack_vecs(
      {&srcs,
       &block_idx_to_tensor_idx,
       &tensor_idx_to_start_tensor_bytes,
       &start_block_idx_per_tensor_chunk,
       &actual_tensor_sizes,
       &pad_tensor_chunk_sizes,
       &num_blocks_per_tensor_chunk},
      device);
  std::vector<int64_t> view_sizes = get_chunk_cat_out_sizes(
      tensors[0].sizes(), dim, num_chunks, chunk_size, dst_elem_size);
  at::native::resize_output(out, view_sizes);
  dim3 blocks(num_blocks_per_chunk, num_chunks, leading_dim);
  dim3 threads(detail::BLOCK_SIZE, 1, 1);
  auto launch = [&](auto kernel) {
    kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        /*srcs=*/reinterpret_cast<src_t**>(packed.second[0]),
        reinterpret_cast<dst_t*>(out.data_ptr()),
        /*block_idx_to_tensor_idx=*/packed.second[1],
        /*tensor_idx_to_start_tensor_bytes=*/packed.second[2],
        /*start_block_idx_per_tensor_chunk=*/packed.second[3],
        /*actual_tensor_sizes=*/packed.second[4],
        /*pad_tensor_chunk_sizes=*/packed.second[5],
        /*num_blocks_per_tensor_chunk=*/packed.second[6],
        slice_size,
        chunk_size,
        dst_elem_size / src_elem_size);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  };
  // Calls whose chunks are all below the threshold launch the kernel without
  // the vector code: its setup alone slows those latency-bound calls down.
  if constexpr (std::is_same_v<dst_t, src_t>) {
    launch(chunk_cat_cuda_kernel<dst_t, src_t, 0>);
  } else if (!std::any_of(
                 pad_tensor_chunk_sizes.begin(),
                 pad_tensor_chunk_sizes.end(),
                 is_vectorized_cast)) {
    launch(chunk_cat_cuda_kernel<dst_t, src_t, 0>);
  } else if (cast_vecs_per_thread == SM107_WIDE_CAST_VECS_PER_THREAD) {
    launch(
        chunk_cat_cuda_kernel<dst_t, src_t, SM107_WIDE_CAST_VECS_PER_THREAD>);
  } else {
    launch(chunk_cat_cuda_kernel<dst_t, src_t, CAST_VECS_PER_THREAD>);
  }
}

} // namespace detail

// See [CUDA fast path for split_with_sizes_copy.out]
void split_with_sizes_copy_out_cuda_contiguous_no_cast(
    const at::Tensor& self,
    at::IntArrayRef split_sizes,
    int64_t dim,
    at::TensorList out) {
  // Every output is empty, and the launch size math below would divide by
  // zero chunks per block.
  if (self.numel() == 0) {
    return;
  }
  const auto device = self.device();
  const auto src_base_addrs =
      detail::get_split_base_addrs(self, split_sizes, dim);
  const auto dst_base_addrs = detail::get_dst_addrs(out);
  const auto src_stride = detail::get_chunk_stride(self, dim);
  const auto split_chunk_sizes =
      detail::get_split_chunk_sizes(self, split_sizes, dim);
  const auto num_chunks = detail::get_num_chunks(self, dim);

  // Calculate the number of blocks required for the first chunk across all
  // splits, assuming each thread only processes BYTES_PER_THREAD bytes.
  int64_t num_blocks = 0;
  for (const auto& split_chunk_size : split_chunk_sizes) {
    num_blocks += detail::div_up(
        split_chunk_size, detail::BLOCK_SIZE * detail::BYTES_PER_THREAD);
  }

  // Calculate the maximum number of blocks to launch. Only consider
  // maxThreadsPerMultiProcessor as a limiting factor as the kernel uses no
  // shared memory and little registers. Over-subscribe the SMs to hide I/O
  // latency.
  const auto num_sms =
      at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  const auto max_threads_per_sm =
      at::cuda::getCurrentDeviceProperties()->maxThreadsPerMultiProcessor;
  const int64_t max_blocks =
      num_sms * max_threads_per_sm / detail::BLOCK_SIZE * 2.0;

  // Make each thread process BYTES_PER_THREAD * iter_factor bytes to regulate
  // block size. Spread iter_factor evenly between chunks_per_block and
  // iters_per_chunk.
  int64_t iter_factor = detail::div_up(num_blocks * num_chunks, max_blocks);
  int64_t chunks_per_block = std::ceil(std::sqrt(iter_factor));
  chunks_per_block = std::min(chunks_per_block, num_chunks);
  const int64_t iters_per_chunk = detail::div_up(iter_factor, chunks_per_block);

  // Launch a logically jagged grid of shape
  // (chunk_size*, num_splits, num_chunks / chunks_per_block)
  // backed by a physical grid of shape
  // (sum(chunk_size), num_chunks / chunks_per_block).
  // A block can find its split_idx via block_idx_to_split_idx.
  std::vector<int64_t> block_idx_to_split_idx;
  std::vector<int64_t> blocks_cumsums{0};
  block_idx_to_split_idx.reserve(num_blocks);
  for (size_t split_idx = 0; split_idx < split_sizes.size(); ++split_idx) {
    const auto blocks = detail::div_up(
        split_chunk_sizes[split_idx],
        detail::BLOCK_SIZE * detail::BYTES_PER_THREAD * iters_per_chunk);
    block_idx_to_split_idx.insert(
        block_idx_to_split_idx.end(), blocks, split_idx);
    blocks_cumsums.push_back(blocks_cumsums.back() + blocks);
  }

  dim3 blocks(blocks_cumsums.back(), num_chunks / chunks_per_block, 1);
  dim3 threads(detail::BLOCK_SIZE, 1, 1);

  auto [_, ptrs] = detail::pack_vecs(
      {&dst_base_addrs,
       &src_base_addrs,
       &split_chunk_sizes,
       &block_idx_to_split_idx,
       &blocks_cumsums},
      device);

  detail::split_with_sizes_copy_out_contiguous_no_cast_kernel<<<
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

void split_with_sizes_copy_out_cuda(
    const Tensor& self,
    IntArrayRef split_sizes,
    int64_t dim,
    TensorList out) {
  bool contiguous_no_cast = self.is_non_overlapping_and_dense();
  for (const auto& t : out) {
    contiguous_no_cast &= t.is_non_overlapping_and_dense();
    contiguous_no_cast &= (t.dtype() == self.dtype());
  }
  if (contiguous_no_cast) {
    // Perform equivalent checks performed by the composite impl
    if (dim < 0) {
      dim = at::maybe_wrap_dim(dim, self.dim());
    }
    TORCH_CHECK(
        self.dim() != 0, "split expects at least a 1-dimensional tensor")

    const int64_t dim_size = self.size(dim);
    int64_t split_sizes_sum = 0;
    for (const auto i : c10::irange(split_sizes.size())) {
      TORCH_CHECK(
          split_sizes[i] >= 0,
          "split_with_sizes expects split_sizes have only non-negative ",
          "entries, but got split_sizes=",
          split_sizes[i]);
      split_sizes_sum += split_sizes[i];
    }
    TORCH_CHECK(
        split_sizes_sum == dim_size,
        "split_with_sizes expects split_sizes to sum exactly to ",
        dim_size,
        " (input tensor's size at dimension ",
        dim,
        "), ",
        "but got split_sizes=",
        split_sizes);

    TORCH_CHECK(
        out.size() == split_sizes.size(),
        "split_with_sizes_copy_out() expected an out= argument of size ",
        split_sizes.size(),
        ", got size ",
        out.size());

    auto out_shape = self.sizes().vec();
    for (const auto i : c10::irange(split_sizes.size())) {
      out_shape[dim] = split_sizes[i];
      if (resize_output_check(out[i], out_shape)) {
        out[i].resize_(out_shape);
      }
      TORCH_CHECK(
          out[i].dtype() == self.dtype(),
          "Expected out tensor to have dtype ",
          self.dtype(),
          ", but got ",
          out[i].dtype(),
          " instead");
      TORCH_CHECK(
          out[i].device() == self.device(),
          "Expected out tensor to have device ",
          self.device(),
          ", but got ",
          out[i].device(),
          " instead");
    }
    split_with_sizes_copy_out_cuda_contiguous_no_cast(
        self, split_sizes, dim, out);
  } else {
    at::native::split_with_sizes_copy_out(self, split_sizes, dim, out);
  }
}

Tensor _chunk_cat_cuda(TensorList tensors, int64_t dim, int64_t num_chunks) {
  dim = at::native::preprocess_chunk_cat_inputs(tensors, dim, num_chunks);
  if (detail::all_contiguous(tensors)) {
    // Return a tensor with the same dtype as input tensors
    int64_t elem_size = tensors[0].element_size();
    int64_t chunk_size =
        detail::get_chunk_size(tensors, dim, num_chunks, elem_size);
    int64_t leading_dim = detail::get_leading_dim(tensors[0].sizes(), dim);
    auto view_sizes = detail::get_chunk_cat_out_sizes(
        tensors[0].sizes(), dim, num_chunks, chunk_size, elem_size);
    Tensor out =
        tensors[0]
            .new_empty(chunk_size * num_chunks * leading_dim / elem_size)
            .view(view_sizes);
    // Type-agnostic copy since out and input tensors have the same type.
    detail::_chunk_cat_out_cuda_contiguous<char, char>(
        tensors, dim, num_chunks, out, elem_size, elem_size);
    return out;
  } else {
    return at::native::_chunk_cat(tensors, dim, num_chunks);
  }
}

Tensor& _chunk_cat_out_cuda(
    TensorList tensors,
    int64_t dim,
    int64_t num_chunks,
    Tensor& out) {
  dim = at::native::preprocess_chunk_cat_inputs(tensors, dim, num_chunks);
  TORCH_CHECK(
      tensors[0].device() == out.device(),
      "_chunk_cat_out_cuda: mismatch between input and out tensor devices");
  bool both_input_output_contiguous =
      detail::all_contiguous(tensors) && out.is_non_overlapping_and_dense();
  if (both_input_output_contiguous &&
      (tensors[0].dtype() == at::ScalarType::BFloat16) &&
      (out.dtype() == at::ScalarType::Float)) {
    // _chunk_cat_out_cuda_contiguous should also support other types, thanks to
    // static_cast_with_inter_type. Here, we dispatch to BFloat16 in and float32
    // out since it is the only known use case.
    detail::_chunk_cat_out_cuda_contiguous<float, BFloat16>(
        tensors,
        dim,
        num_chunks,
        out,
        out.element_size(),
        tensors[0].element_size());
  } else if (
      both_input_output_contiguous && tensors[0].dtype() == out.dtype()) {
    // Type-agnostic copy since out and input tensors have the same type.
    detail::_chunk_cat_out_cuda_contiguous<char, char>(
        tensors,
        dim,
        num_chunks,
        out,
        out.element_size(),
        tensors[0].element_size());
  } else {
    at::native::_chunk_cat_out(tensors, dim, num_chunks, out);
  }
  return out;
}

} // namespace at::native
