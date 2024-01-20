#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/Dispatch.h>
#include <ATen/core/Tensor.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/ops/empty.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/Logging.h>

namespace at::native {

static constexpr int64_t BLOCK_SIZE = 128;
static constexpr int64_t BYTES_PER_THREAD = 16;

namespace detail {

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
  for (size_t i = 0; i < BYTES_PER_THREAD / sizeof(T); ++i) {
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
      dst, chunk_size, BYTES_PER_THREAD, align_off, aligned_size);

  for (int64_t off = align_off + thread_idx * BYTES_PER_THREAD;
       off < align_off + aligned_size;
       off += num_threads * BYTES_PER_THREAD) {
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
  const auto* data_ptr = static_cast<char*>(tensor.data_ptr());
  const auto strides = tensor.strides();
  const auto element_sz = tensor.element_size();
  int64_t off = 0;
  std::vector<int64_t> split_base_addrs;
  for (const auto& split_size : split_sizes) {
    split_base_addrs.push_back(reinterpret_cast<int64_t>(data_ptr + off));
    off += split_size * strides[dim] * element_sz;
  }
  return split_base_addrs;
}

static inline std::vector<int64_t> get_dst_addrs(at::TensorList out) {
  std::vector<int64_t> addrs;
  for (const auto& tensor : out) {
    addrs.push_back(reinterpret_cast<int64_t>(tensor.data_ptr()));
  }
  return addrs;
}

// Caculate the chunk size for each split in bytes.
static inline std::vector<int64_t> get_split_chunk_sizes(
    const at::Tensor& tensor,
    at::IntArrayRef split_sizes,
    int64_t dim) {
  const auto stride = tensor.stride(dim);
  const auto element_sz = tensor.element_size();
  std::vector<int64_t> split_chunk_sizes;
  for (const auto& split_size : split_sizes) {
    split_chunk_sizes.push_back(split_size * stride * element_sz);
  }
  return split_chunk_sizes;
}

// Caculate the chunk stride in bytes. This is the same for all splits.
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
  offset = 0;
  for (const auto* vec : vecs) {
    ptrs.push_back(packed.data_ptr<int64_t>() + offset);
    offset += vec->size();
  }
  return std::make_pair(packed, ptrs);
}

} // namespace detail

// A fast path for contiguous operands requiring no cast.
//
// Since the operands are contiguous, each src split consists of a fixed number
// of contiguous chunks (depending on the split dim), separated by a fixed
// stride. Each dst split consists of the same number of contiguous chunks,
// with no gaps in-between. This function calculates the base addresses, chunk
// sizes, and stride for both src and dst splits, and copies every chunk pair
// with an efficient copy kernel.
void split_with_sizes_copy_out_cuda_contiguous_no_cast(
    const at::Tensor& self,
    at::IntArrayRef split_sizes,
    int64_t dim,
    at::TensorList out) {
  TORCH_CHECK(self.dim() != 0, "split expects at least a 1-dimensional tensor")
  const int64_t dim_size = self.size(dim);
  int64_t split_sizes_sum = 0;
  auto sizes = self.sizes().vec();
  for (const auto i : c10::irange(split_sizes.size())) {
    TORCH_CHECK(
        split_sizes[i] >= 0,
        "split_with_sizes expects split_sizes have only non-negative ",
        "entries, but got split_sizes=",
        split_sizes[i]);
    split_sizes_sum += split_sizes[i];
    sizes[dim] = split_sizes[i];
    TORCH_CHECK(
        ArrayRef(sizes) == out[i].sizes(),
        "shape mismatch between the splits and the outputs");
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
    num_blocks +=
        detail::div_up(split_chunk_size, BLOCK_SIZE * BYTES_PER_THREAD);
  }

  // Calculate the maximum number of blocks to launch. Only consider
  // maxThreadsPerMultiProcessor as a limiting factor as the kernel uses no
  // shared memory and little registers. Over-subscribe the SMs to hide I/O
  // latency.
  const auto num_sms =
      at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  const auto max_threads_per_sm =
      at::cuda::getCurrentDeviceProperties()->maxThreadsPerMultiProcessor;
  const int64_t max_blocks = num_sms * max_threads_per_sm / BLOCK_SIZE * 2.0;

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
        BLOCK_SIZE * BYTES_PER_THREAD * iters_per_chunk);
    block_idx_to_split_idx.insert(
        block_idx_to_split_idx.end(), blocks, split_idx);
    blocks_cumsums.push_back(blocks_cumsums.back() + blocks);
  }

  dim3 blocks(blocks_cumsums.back(), num_chunks / chunks_per_block, 1);
  dim3 threads(BLOCK_SIZE, 1, 1);

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

  for (const auto& t : out) {
    t.unsafeGetTensorImpl()->bump_version();
  }
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
    split_with_sizes_copy_out_cuda_contiguous_no_cast(
        self, split_sizes, dim, out);
  } else {
    auto tmp = self.split_with_sizes(split_sizes, dim);
    for (const auto i : c10::irange(out.size())) {
      out[i].copy_(tmp[i]);
    }
  }
}

} // namespace at::native
