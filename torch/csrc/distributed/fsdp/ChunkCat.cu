#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/native/Resize.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/TypeCast.h>
#include <c10/util/accumulate.h>
#include <c10/util/irange.h>
#include <torch/csrc/distributed/fsdp/ChunkCat.h>
#include <torch/csrc/distributed/fsdp/CollectiveCopy.hpp>
#include <torch/library.h>

#include <algorithm>
#include <cmath>

namespace torch::distributed::fsdp {
namespace {

// The device code and pack_vecs are copied verbatim from
// aten/src/ATen/native/cuda/TensorShape.cu, where they are not exported, apart
// from namespace qualification.
static constexpr int64_t BLOCK_SIZE = 128;
static constexpr int64_t BYTES_PER_THREAD = 16;
static constexpr int64_t BYTES_PER_BLOCK = BYTES_PER_THREAD * BLOCK_SIZE;

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
  align_off = div_up(ptr_val, alignment) * alignment - ptr_val;
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
    // Opportunistically vectorize reads
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

// Copy `max_chunk_size` bytes from `src` to `dst` by `num_threads`, and pad
// zero when `src` size (i.e., actual_chunk_size) is less than `max_chunk_size`.
// Assume elements of src and dst have the same data type.
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
          c10::static_cast_with_inter_type<dst_t, src_t>::apply(
              src_ptr[elem_index]);
      elem_index += num_threads;
    }
    while (elem_index < max_num_elems) {
      dst_ptr[elem_index] =
          c10::static_cast_with_inter_type<dst_t, int>::apply(0);
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

template <typename dst_t, typename src_t>
void launch_chunk_cat(
    const std::vector<int64_t*>& ptrs,
    int64_t* block_idx_to_tensor_idx,
    int64_t num_blocks,
    int64_t num_chunks,
    int64_t chunk_size,
    at::Tensor& out) {
  chunk_cat_cuda_kernel<dst_t, src_t>
      <<<dim3(num_blocks, num_chunks, 1),
         dim3(BLOCK_SIZE, 1, 1),
         0,
         at::cuda::getCurrentCUDAStream()>>>(
          reinterpret_cast<src_t**>(ptrs[0]),
          static_cast<dst_t*>(out.mutable_data_ptr()),
          block_idx_to_tensor_idx,
          ptrs[2],
          ptrs[3],
          ptrs[4],
          ptrs[5],
          ptrs[6],
          /*slice_size=*/num_chunks * chunk_size,
          chunk_size,
          /*dst_to_src_ratio=*/sizeof(dst_t) / sizeof(src_t));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Inputs already in out's dtype are copied by one launch, and cast_dtype inputs
// into an fp32 out are cast by a second one. The output layout doesn't depend
// on input dtypes, so both launches share one metadata upload and differ only
// in their block_idx_to_tensor_idx ranges.
// start_block_idx_per_tensor_chunk is relative to each tensor's launch.
// Each index into the first num_leading_dims[i] dims of tensors[i] is a
// separate _chunk_cat input. Empty num_leading_dims means all zeros.
void fused_chunk_cat(
    at::TensorList tensors,
    at::IntArrayRef num_leading_dims,
    int64_t num_chunks,
    at::ScalarType cast_dtype,
    at::Tensor& out) {
  c10::cuda::CUDAGuard device_guard(out.device());
  const auto out_dtype = out.scalar_type();
  const auto num_tensors = tensors.size();
  std::vector<int64_t> srcs;
  std::vector<int64_t> copy_blocks;
  std::vector<int64_t> cast_blocks;
  std::vector<int64_t> tensor_idx_to_start_tensor_bytes;
  std::vector<int64_t> start_block_idx_per_tensor_chunk;
  std::vector<int64_t> actual_tensor_sizes;
  std::vector<int64_t> pad_tensor_chunk_sizes;
  std::vector<int64_t> num_blocks_per_tensor_chunk;
  srcs.reserve(num_tensors);
  tensor_idx_to_start_tensor_bytes.reserve(num_tensors);
  start_block_idx_per_tensor_chunk.reserve(num_tensors);
  actual_tensor_sizes.reserve(num_tensors);
  pad_tensor_chunk_sizes.reserve(num_tensors);
  num_blocks_per_tensor_chunk.reserve(num_tensors);
  int64_t chunk_size = 0;
  for (const auto i : c10::irange(num_tensors)) {
    const at::Tensor& tensor = tensors[i];
    const auto sizes = tensor.sizes();
    const int64_t dim = num_leading_dims.empty() ? 0 : num_leading_dims[i];
    const int64_t row_numel = c10::multiply_integers(sizes.slice(dim + 1));
    const int64_t pad_tensor_chunk_size =
        div_up(sizes[dim], num_chunks) * row_numel * out.element_size();
    const int64_t actual_tensor_size =
        sizes[dim] * row_numel * tensor.element_size();
    const int64_t num_blocks = div_up(pad_tensor_chunk_size, BYTES_PER_BLOCK);
    auto& blocks =
        tensor.scalar_type() == out_dtype ? copy_blocks : cast_blocks;
    // The cast launch below reads cast_dtype and writes fp32
    TORCH_INTERNAL_ASSERT(
        tensor.scalar_type() == out_dtype ||
        (tensor.scalar_type() == cast_dtype && out_dtype == at::kFloat));
    const auto src = reinterpret_cast<int64_t>(tensor.const_data_ptr());
    const int64_t num_slices = c10::multiply_integers(sizes.slice(0, dim));
    for (const auto slice : c10::irange(num_slices)) {
      srcs.push_back(src + slice * actual_tensor_size);
      tensor_idx_to_start_tensor_bytes.push_back(chunk_size);
      start_block_idx_per_tensor_chunk.push_back(
          static_cast<int64_t>(blocks.size()));
      blocks.insert(
          blocks.end(), num_blocks, static_cast<int64_t>(srcs.size()) - 1);
      actual_tensor_sizes.push_back(actual_tensor_size);
      pad_tensor_chunk_sizes.push_back(pad_tensor_chunk_size);
      num_blocks_per_tensor_chunk.push_back(num_blocks);
      chunk_size += pad_tensor_chunk_size;
    }
  }
  const auto num_copy_blocks = static_cast<int64_t>(copy_blocks.size());
  const auto num_cast_blocks = static_cast<int64_t>(cast_blocks.size());
  auto& block_idx_to_tensor_idx = copy_blocks;
  block_idx_to_tensor_idx.insert(
      block_idx_to_tensor_idx.end(), cast_blocks.begin(), cast_blocks.end());
  at::native::resize_output(out, {num_chunks, chunk_size / out.element_size()});
  // `packed` keeps the device metadata alive until the kernels are enqueued.
  auto [packed, ptrs] = pack_vecs(
      {&srcs,
       &block_idx_to_tensor_idx,
       &tensor_idx_to_start_tensor_bytes,
       &start_block_idx_per_tensor_chunk,
       &actual_tensor_sizes,
       &pad_tensor_chunk_sizes,
       &num_blocks_per_tensor_chunk},
      out.device());
  if (num_copy_blocks > 0) {
    launch_chunk_cat<char, char>(
        ptrs, ptrs[1], num_copy_blocks, num_chunks, chunk_size, out);
  }
  if (num_cast_blocks > 0) {
    AT_DISPATCH_REDUCED_FLOATING_TYPES(cast_dtype, "fused_chunk_cat", [&] {
      launch_chunk_cat<float, scalar_t>(
          ptrs,
          ptrs[1] + num_copy_blocks,
          num_cast_blocks,
          num_chunks,
          chunk_size,
          out);
    });
  }
}

// Same-dtype groups take the composite, which forwards them to _chunk_cat.
// dim != 0, which FSDP doesn't use, also takes the composite.
void chunk_cat_mixed_dtype_cuda(
    at::TensorList tensors,
    int64_t dim,
    int64_t num_chunks,
    at::Tensor& out) {
  const auto out_dtype = out.scalar_type();
  const bool mixed_dtypes =
      std::any_of(tensors.begin(), tensors.end(), [&](const at::Tensor& t) {
        return t.scalar_type() != tensors[0].scalar_type();
      });
  // TODO: Also cast fp16 inputs into an fp32 out here, uniform groups included.
  // fp16 + fp32 groups take the composite's copies, and uniform fp16 takes
  // _chunk_cat's copy per input; only bf16 compute is a known use case so far.
  const bool use_fused_kernel =
      mixed_dtypes && dim == 0 && num_chunks >= 1 && out.is_contiguous() &&
      std::all_of(tensors.begin(), tensors.end(), [&](const at::Tensor& t) {
        return t.dim() > 0 && t.numel() > 0 && t.device() == out.device() &&
            t.is_contiguous() &&
            (t.scalar_type() == out_dtype ||
             (t.scalar_type() == at::kBFloat16 && out_dtype == at::kFloat));
      });
  if (!use_fused_kernel) {
    chunk_cat_mixed_dtype(tensors, dim, num_chunks, out);
    return;
  }
  fused_chunk_cat(
      tensors, /*num_leading_dims=*/{}, num_chunks, at::kBFloat16, out);
}

// The fused kernels copy raw bytes of contiguous tensors on one device, so
// other tensors, conj or neg views, and overlapping memory take the composite.
bool use_composite(at::TensorList tensors, const at::Tensor& other) {
  std::vector<std::pair<uintptr_t, uintptr_t>> ranges;
  ranges.reserve(tensors.size() + 1);
  const auto add = [&](const at::Tensor& t) {
    if (!t.is_cuda() || t.device() != other.device() || !t.is_contiguous() ||
        t.is_conj() || t.is_neg()) {
      return false;
    }
    if (t.numel() > 0) {
      const auto begin = reinterpret_cast<uintptr_t>(t.const_data_ptr());
      ranges.emplace_back(begin, begin + t.nbytes());
    }
    return true;
  };
  if (!add(other) || !std::all_of(tensors.begin(), tensors.end(), add)) {
    return true;
  }
  std::sort(ranges.begin(), ranges.end());
  return std::adjacent_find(
             ranges.begin(), ranges.end(), [](const auto& a, const auto& b) {
               return b.first < a.second;
             }) != ranges.end();
}

void all_gather_copy_out_cuda(
    at::TensorList out,
    const at::Tensor& input,
    c10::SymIntArrayRef split_sizes,
    c10::SymIntArrayRef outer_sizes,
    int64_t num_chunks) {
  if (use_composite(out, input)) {
    all_gather_copy_out(out, input, split_sizes, outer_sizes, num_chunks);
    return;
  }
  check_all_gather_copy_out_inputs(
      out, input, split_sizes, outer_sizes, num_chunks);
  c10::cuda::CUDAGuard device_guard(input.device());
  std::vector<int64_t> srcs;
  std::vector<int64_t> dsts;
  std::vector<int64_t> chunk_sizes;
  auto src = reinterpret_cast<int64_t>(input.const_data_ptr());
  const auto elem_size = input.element_size();
  for (const auto i : c10::irange(out.size())) {
    const auto split_size = split_sizes[i].expect_int();
    if (split_size == 0) {
      continue;
    }
    const auto outer_size = outer_sizes[i].expect_int();
    const int64_t chunk_size = split_size / outer_size * elem_size;
    const auto dst = reinterpret_cast<int64_t>(out[i].mutable_data_ptr());
    for (const auto outer_idx : c10::irange(outer_size)) {
      srcs.push_back(src + outer_idx * chunk_size);
      dsts.push_back(dst + outer_idx * num_chunks * chunk_size);
      chunk_sizes.push_back(chunk_size);
    }
    src += split_size * elem_size;
  }
  if (srcs.empty()) {
    return;
  }
  int64_t num_blocks = 0;
  for (const auto chunk_size : chunk_sizes) {
    num_blocks += div_up(chunk_size, BYTES_PER_BLOCK);
  }
  const auto* properties = at::cuda::getCurrentDeviceProperties();
  const int64_t max_blocks =
      static_cast<int64_t>(properties->multiProcessorCount) *
      properties->maxThreadsPerMultiProcessor / BLOCK_SIZE * 2;
  const int64_t iter_factor = div_up(num_blocks * num_chunks, max_blocks);
  int64_t chunks_per_block = std::ceil(std::sqrt(iter_factor));
  chunks_per_block = std::min(chunks_per_block, num_chunks);
  const int64_t iters_per_chunk = div_up(iter_factor, chunks_per_block);
  std::vector<int64_t> block_idx_to_split_idx;
  std::vector<int64_t> blocks_cumsums{0};
  block_idx_to_split_idx.reserve(num_blocks);
  for (const auto i : c10::irange(chunk_sizes.size())) {
    const auto blocks =
        div_up(chunk_sizes[i], BYTES_PER_BLOCK * iters_per_chunk);
    block_idx_to_split_idx.insert(
        block_idx_to_split_idx.end(), blocks, static_cast<int64_t>(i));
    blocks_cumsums.push_back(blocks_cumsums.back() + blocks);
  }
  // `packed` keeps the device metadata alive until the kernel is enqueued.
  auto [packed, ptrs] = pack_vecs(
      {&dsts, &srcs, &chunk_sizes, &block_idx_to_split_idx, &blocks_cumsums},
      input.device());
  split_with_sizes_copy_out_contiguous_no_cast_kernel<<<
      dim3(blocks_cumsums.back(), num_chunks / chunks_per_block, 1),
      dim3(BLOCK_SIZE, 1, 1),
      0,
      at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<char**>(ptrs[0]),
      reinterpret_cast<char**>(ptrs[1]),
      ptrs[2],
      ptrs[3],
      ptrs[4],
      /*src_stride=*/input.numel() / num_chunks * elem_size,
      num_chunks);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

at::Tensor& reduce_scatter_copy_in_cuda(
    at::Tensor& out,
    at::TensorList tensors,
    at::IntArrayRef num_leading_dims,
    int64_t num_chunks) {
  const auto out_dtype = out.scalar_type();
  const auto dtype = tensors.empty() ? out_dtype : tensors[0].scalar_type();
  const bool fused_dtypes = dtype == out_dtype ||
      (out_dtype == at::kFloat &&
       (dtype == at::kBFloat16 || dtype == at::kHalf));
  if (!fused_dtypes || use_composite(tensors, out)) {
    return reduce_scatter_copy_in(out, tensors, num_leading_dims, num_chunks);
  }
  check_reduce_scatter_copy_in_inputs(
      out, tensors, num_leading_dims, num_chunks);
  auto chunks = out.view({num_chunks, -1});
  fused_chunk_cat(tensors, num_leading_dims, num_chunks, dtype, chunks);
  return out;
}

TORCH_LIBRARY_IMPL(fsdp, CUDA, m) {
  m.impl("chunk_cat_mixed_dtype", TORCH_FN(chunk_cat_mixed_dtype_cuda));
  m.impl("_all_gather_copy_out_", TORCH_FN(all_gather_copy_out_cuda));
  m.impl("_reduce_scatter_copy_in_", TORCH_FN(reduce_scatter_copy_in_cuda));
}

} // namespace
} // namespace torch::distributed::fsdp
