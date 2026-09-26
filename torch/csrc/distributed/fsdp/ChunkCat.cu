#include <ATen/cuda/CUDAContext.h>
#include <ATen/native/Resize.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/TypeCast.h>
#include <c10/util/irange.h>
#include <torch/csrc/distributed/fsdp/ChunkCat.h>
#include <torch/library.h>

namespace torch::distributed::fsdp {
namespace {

// The device code and pack_vecs are copied from
// aten/src/ATen/native/cuda/TensorShape.cu, where they are not exported. See
// NOTE [CUDA kernel for chunk_cat] there.
constexpr int64_t BLOCK_SIZE = 128;
constexpr int64_t BYTES_PER_THREAD = 16;
constexpr int64_t BYTES_PER_BLOCK = BYTES_PER_THREAD * BLOCK_SIZE;

__host__ __device__ inline int64_t div_up(int64_t a, int64_t b) {
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
__device__ inline bool is_aligned(const void* addr) {
  return reinterpret_cast<uintptr_t>(addr) % sizeof(T) == 0;
}

__device__ __inline__ void get_aligned_region(
    char* ptr,
    const int64_t chunk_size,
    const int64_t alignment,
    int64_t& align_off,
    int64_t& aligned_size) {
  const int64_t ptr_val = reinterpret_cast<uintptr_t>(ptr);
  align_off = div_up(ptr_val, alignment) * alignment - ptr_val;
  aligned_size = (chunk_size - align_off) / alignment * alignment;
}

template <typename dst_t, typename src_t>
__device__ __inline__ void copy_chunk_with_pad(
    dst_t* dst_ptr,
    src_t* src_ptr,
    int64_t max_chunk_size,
    int64_t actual_chunk_size,
    int64_t thread_idx,
    int64_t num_threads) {
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
      dst_ptr[elem_index] = c10::static_cast_with_inter_type<dst_t, int>::apply(0);
      elem_index += num_threads;
    }
    return;
  }
  char* dst = reinterpret_cast<char*>(dst_ptr);
  char* src = reinterpret_cast<char*>(src_ptr);
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
  if (thread_idx < align_off && thread_idx < max_chunk_size) {
    char val = (char)0;
    if (thread_idx < actual_chunk_size) {
      val = src[thread_idx];
    }
    dst[thread_idx] = val;
  }
  while (align_end + thread_idx < max_chunk_size) {
    char val = (char)0;
    if (align_end + thread_idx < actual_chunk_size) {
      val = src[align_end + thread_idx];
    }
    dst[align_end + thread_idx] = val;
    align_end += num_threads;
  }
}

template <typename dst_t, typename src_t>
__global__ void chunk_cat_cuda_kernel(
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
  const int64_t num_threads =
      num_blocks_per_tensor_chunk[tensor_idx] * BLOCK_SIZE;
  const int64_t thread_idx = tile_idx * BLOCK_SIZE + threadIdx.x;
  char* src_addr = reinterpret_cast<char**>(src)[tensor_idx] +
      slice_idx * actual_tensor_sizes[tensor_idx] +
      chunk_idx * pad_tensor_chunk_sizes[tensor_idx] / dst_to_src_ratio;
  char* dst_addr = reinterpret_cast<char*>(dst) + slice_idx * slice_size +
      chunk_idx * chunk_size + tensor_idx_to_start_tensor_bytes[tensor_idx];
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

// Same-dtype groups take the composite, which forwards them to _chunk_cat.
// dim != 0, which FSDP doesn't use, also takes the composite. In mixed groups,
// inputs already in out's dtype are copied by one launch and bf16 inputs into
// an fp32 out are cast by a second one. The output layout doesn't depend on
// input dtypes, so both launches share one metadata upload and differ only in
// their block_idx_to_tensor_idx ranges.
// start_block_idx_per_tensor_chunk is relative to each tensor's launch.
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
  const bool use_fused_kernel = mixed_dtypes && dim == 0 && num_chunks >= 1 &&
      out.is_contiguous() &&
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
  c10::cuda::CUDAGuard device_guard(out.device());
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
    const int64_t pad_tensor_chunk_size = div_up(tensor.size(0), num_chunks) *
        (tensor.numel() / tensor.size(0)) * out.element_size();
    const int64_t num_blocks = div_up(pad_tensor_chunk_size, BYTES_PER_BLOCK);
    auto& blocks =
        tensor.scalar_type() == out_dtype ? copy_blocks : cast_blocks;
    srcs.push_back(reinterpret_cast<int64_t>(tensor.const_data_ptr()));
    tensor_idx_to_start_tensor_bytes.push_back(chunk_size);
    start_block_idx_per_tensor_chunk.push_back(
        static_cast<int64_t>(blocks.size()));
    blocks.insert(blocks.end(), num_blocks, static_cast<int64_t>(i));
    actual_tensor_sizes.push_back(static_cast<int64_t>(tensor.nbytes()));
    pad_tensor_chunk_sizes.push_back(pad_tensor_chunk_size);
    num_blocks_per_tensor_chunk.push_back(num_blocks);
    chunk_size += pad_tensor_chunk_size;
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
    launch_chunk_cat<float, at::BFloat16>(
        ptrs,
        ptrs[1] + num_copy_blocks,
        num_cast_blocks,
        num_chunks,
        chunk_size,
        out);
  }
}

TORCH_LIBRARY_IMPL(fsdp, CUDA, m) {
  m.impl("chunk_cat_mixed_dtype", TORCH_FN(chunk_cat_mixed_dtype_cuda));
}

} // namespace
} // namespace torch::distributed::fsdp
