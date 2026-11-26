#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/detail/OffsetCalculator.cuh>
#include <ATen/detail/CUDAHooksInterface.h>
#include <c10/cuda/CUDAMacros.h>
#include <ATen/native/cuda/IndexKernelUtils.h>
#include <ATen/native/cuda/MemoryAccess.cuh>
#include <ATen/cuda/Atomic.cuh>

#include <c10/macros/Macros.h>
#include <c10/util/Exception.h>
#include <c10/util/Half.h>
#include <c10/util/BFloat16.h>
#include <c10/util/CallOnce.h>
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/ceil_div.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <initializer_list>
#include <tuple>
#include <utility>
#include <type_traits>

namespace at::native {
template <int Alignment, typename index_t>
__global__ void vectorized_gather_kernel(char * out, char * inp, index_t * idx, int num_ind, int64_t slice_size, int64_t ind_dim_size, int64_t inp_stride, int64_t out_stride, bool allow_neg_indices) {
    for (auto entry = blockIdx.x * blockDim.y + threadIdx.y; entry < static_cast<unsigned int>(num_ind); entry += gridDim.x * blockDim.y) {
      int64_t ind = idx[entry];
      if (allow_neg_indices) {
          ind = (ind < 0) ? ind + ind_dim_size : ind;
      }
      CUDA_KERNEL_ASSERT_VERBOSE(ind >=0 && ind < ind_dim_size && "vectorized gather kernel index out of bounds", "Expected 0 <= index < ind_dim_size(%ld), but got index = %ld", ind_dim_size, ind);
      // Keep the final byte-offset increment unsigned, as in vectorized scatter.
      for (uint32_t off = (blockDim.x * blockIdx.y + threadIdx.x) * Alignment; off < slice_size; off += blockDim.x * gridDim.y * Alignment) {
        auto vec = at::native::memory::ld_vec<Alignment>(inp + ind * inp_stride + off);
        at::native::memory::st_vec<Alignment>(out + entry * (int32_t)out_stride + off, vec);  // out offset is guaranteed to be within int32 limits
      }
    }
}



// Launch limits shared by vectorized_gather_kernel_launch and the gfx1250
// strided-index launch.
constexpr int64_t kVectorizedGatherMaxThreads = 256;
// Packing wider rows regressed Hopper; leave them one row per block.
constexpr int64_t kVectorizedGatherMaxPackedThreads = 32;
constexpr int64_t kVectorizedGatherMaxBlockWaves = 32;

#ifdef USE_ROCM
bool is_gfx1250_device(c10::DeviceIndex device_index) {
  TORCH_INTERNAL_ASSERT(device_index >= 0 && device_index < C10_COMPILE_TIME_MAX_GPUS);
  static std::array<c10::once_flag, C10_COMPILE_TIME_MAX_GPUS> once_flags;
  static std::array<bool, C10_COMPILE_TIME_MAX_GPUS> results{};
  c10::call_once(once_flags[device_index], [device_index] {
    results[device_index] = at::detail::getCUDAHooks().isGPUArch({"gfx1250"}, device_index);
  });
  return results[device_index];
}

// Row-sized launch of the vectorized gathers, used on gfx1250 (wave32), which
// dispatches many short-lived multi-wave blocks slowly: each row gets one block
// sized to the row instead of a full max_threads block, with no column split.

// Vectors copied per thread in the row-sized launch.
constexpr int64_t kRowSizedGatherVectorsPerThread = 8;
// Fewest rows per CU for the row-sized launch; with fewer rows the default
// launch's column split (grid.y) keeps more threads busy.
constexpr int64_t kRowSizedGatherMinRowsPerCU = 4;
// Fewest 16-byte vectors per CU for the row-sized launch; with less work its
// small row-sized blocks leave CUs idle that the default launch's 256-thread
// blocks keep busy.
constexpr int64_t kRowSizedGatherMinVectorsPerCU = 6144;

static inline bool can_use_row_sized_vectorized_gather(const cudaDeviceProp& props, int64_t num_rows, int64_t vectors_per_row) {
  const int64_t num_cus = props.multiProcessorCount;
  return num_rows >= kRowSizedGatherMinRowsPerCU * num_cus && num_rows * vectors_per_row >= kRowSizedGatherMinVectorsPerCU * num_cus;
}

// threads per row = ceil(vectors_per_row / kRowSizedGatherVectorsPerThread),
// rounded up to whole waves and capped at kVectorizedGatherMaxThreads (wider
// rows loop); one row per block, grid.y = 1; blocks = rows, capped at the
// default launch's budget of kVectorizedGatherMaxBlockWaves resident sets of
// blocks.
static inline std::tuple<dim3, dim3> get_row_sized_gather_grid(const cudaDeviceProp& props, int64_t vectors_per_row, int64_t num_rows) {
  const int64_t threads_per_row = std::min(
      kVectorizedGatherMaxThreads, at::round_up(at::ceil_div(vectors_per_row, kRowSizedGatherVectorsPerThread), static_cast<int64_t>(props.warpSize)));
  const int64_t max_blocks = static_cast<int64_t>(props.multiProcessorCount) * (props.maxThreadsPerMultiProcessor / threads_per_row) * kVectorizedGatherMaxBlockWaves;
  return {dim3(static_cast<uint32_t>(std::min(num_rows, max_blocks))), dim3(static_cast<uint32_t>(threads_per_row))};
}
#endif

template <int64_t Alignment, typename index_t>
void vectorized_gather_kernel_launch(char * out, char * inp, index_t * idx, int num_ind,
                                     int64_t slice_size_in_bytes, int64_t ind_dim_size, int64_t inp_stride_bytes, int64_t out_stride_bytes, bool allow_neg_indices){

  constexpr int64_t max_num_threads = kVectorizedGatherMaxThreads;
  constexpr int64_t max_packed_num_threads = kVectorizedGatherMaxPackedThreads;
  constexpr int64_t max_block_waves = kVectorizedGatherMaxBlockWaves;
  auto num_vectors = at::ceil_div(slice_size_in_bytes, Alignment);
  const bool is_packed = num_vectors <= max_packed_num_threads;
  const auto stream = at::cuda::getCurrentCUDAStream();
#ifdef USE_ROCM
  if (!is_packed && is_gfx1250_device(stream.device_index())) {
    const auto& props = *at::cuda::getDeviceProperties(stream.device_index());
    if (can_use_row_sized_vectorized_gather(props, num_ind, num_vectors)) {
      const auto [grid, block] = get_row_sized_gather_grid(props, num_vectors, num_ind);
      vectorized_gather_kernel<Alignment, index_t><<<grid, block, 0, stream>>>(out, inp, idx, num_ind, slice_size_in_bytes,
      ind_dim_size, inp_stride_bytes, out_stride_bytes, allow_neg_indices);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
      return;
    }
  }
#endif
  // x copies vectors within a row; y packs rows without rounding each to a warp.
  auto num_threads = is_packed ? num_vectors : std::min(max_num_threads, at::round_up(num_vectors, static_cast<int64_t>(at::cuda::warp_size())));
  auto entries_per_block = is_packed ? max_num_threads / num_vectors : 1;
  const auto* properties = at::cuda::getCurrentDeviceProperties();
  // A partially filled warp still consumes all of its thread slots.
  const auto warp_threads_per_block = at::round_up(num_threads * entries_per_block, static_cast<int64_t>(at::cuda::warp_size()));
  const int64_t resident_blocks = properties->multiProcessorCount * (properties->maxThreadsPerMultiProcessor / warp_threads_per_block);
  // Cap rows to 32 thread-limited waves; scattered reads need more concurrency
  // than ScatterCopyOp's random writes.
  const int64_t max_blocks = resident_blocks * max_block_waves;
  const auto row_blocks = at::ceil_div(static_cast<int64_t>(num_ind), entries_per_block);
  const auto grid_x = std::min(row_blocks, max_blocks);
  uint32_t grid_y = properties->maxGridSize[1];
  grid_y = std::min(static_cast<uint32_t>(at::ceil_div(slice_size_in_bytes, max_num_threads * Alignment)), grid_y);
  // Budget the full grid; few very wide rows still need column parallelism.
  grid_y = std::min(static_cast<int64_t>(grid_y), max_blocks / grid_x);
  dim3 grid = {static_cast<uint32_t>(grid_x), grid_y, 1};
  dim3 block = {static_cast<uint32_t>(num_threads), static_cast<uint32_t>(entries_per_block), 1};
  vectorized_gather_kernel<Alignment, index_t><<<grid, block, 0, stream>>>(out, inp, idx, num_ind, slice_size_in_bytes,
  ind_dim_size, inp_stride_bytes, out_stride_bytes, allow_neg_indices);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// explicit template instantiation
template void vectorized_gather_kernel_launch<16, int64_t>(char * out, char * inp, int64_t * idx, int num_ind, int64_t slice_size_in_bytes,
int64_t ind_dim_size, int64_t inp_stride_bytes, int64_t out_stride_bytes, bool allow_neg_indices);
template void vectorized_gather_kernel_launch<16, int32_t>(char * out, char * inp, int32_t * idx, int num_ind, int64_t slice_size_in_bytes,
int64_t ind_dim_size, int64_t inp_stride_bytes, int64_t out_stride_bytes, bool allow_neg_indices);

#ifdef USE_ROCM
namespace {
// vectorized_gather_kernel for an index that is strided along the gathered
// dim: entry i reads idx[i * idx_stride].
template <int Alignment, typename index_t>
C10_LAUNCH_BOUNDS_1(kVectorizedGatherMaxThreads)
__global__ void vectorized_gather_strided_index_kernel(char * out, char * inp, index_t * idx, int64_t idx_stride, int num_ind, int64_t slice_size, int64_t ind_dim_size, int64_t inp_stride, int64_t out_stride) {
    for (auto entry = blockIdx.x * blockDim.y + threadIdx.y; entry < static_cast<unsigned int>(num_ind); entry += gridDim.x * blockDim.y) {
      int64_t ind = idx[entry * idx_stride];
      CUDA_KERNEL_ASSERT_VERBOSE(ind >=0 && ind < ind_dim_size && "vectorized gather kernel index out of bounds", "Expected 0 <= index < ind_dim_size(%ld), but got index = %ld", ind_dim_size, ind);
      for (uint32_t off = (blockDim.x * blockIdx.y + threadIdx.x) * Alignment; off < slice_size; off += blockDim.x * gridDim.y * Alignment) {
        auto vec = at::native::memory::ld_vec<Alignment>(inp + ind * inp_stride + off);
        at::native::memory::st_vec<Alignment>(out + entry * (int32_t)out_stride + off, vec);
      }
    }
}
} // namespace

template <int64_t Alignment>
bool can_use_vectorized_gather_strided_index(c10::DeviceIndex device, int64_t slice_size_in_bytes, int64_t num_ind) {
  // Packed rows (several per block in vectorized_gather_kernel_launch) do not
  // take the row-sized launch.
  const auto num_vectors = at::ceil_div(slice_size_in_bytes, Alignment);
  return num_vectors > kVectorizedGatherMaxPackedThreads && can_use_row_sized_vectorized_gather(*at::cuda::getDeviceProperties(device), num_ind, num_vectors);
}

template <int64_t Alignment, typename index_t>
void launch_vectorized_gather_strided_index_kernel(char * out, char * inp, index_t * idx, int64_t idx_stride, int num_ind,
                                                   int64_t slice_size_in_bytes, int64_t ind_dim_size, int64_t inp_stride_bytes, int64_t out_stride_bytes) {
  const auto stream = at::cuda::getCurrentCUDAStream();
  const auto [grid, block] = get_row_sized_gather_grid(*at::cuda::getDeviceProperties(stream.device_index()), at::ceil_div(slice_size_in_bytes, Alignment), num_ind);
  vectorized_gather_strided_index_kernel<Alignment, index_t><<<grid, block, 0, stream>>>(
      out, inp, idx, idx_stride, num_ind, slice_size_in_bytes, ind_dim_size, inp_stride_bytes, out_stride_bytes);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template bool can_use_vectorized_gather_strided_index<16>(c10::DeviceIndex device, int64_t slice_size_in_bytes, int64_t num_ind);
template void launch_vectorized_gather_strided_index_kernel<16, int64_t>(char * out, char * inp, int64_t * idx, int64_t idx_stride, int num_ind,
int64_t slice_size_in_bytes, int64_t ind_dim_size, int64_t inp_stride_bytes, int64_t out_stride_bytes);
template void launch_vectorized_gather_strided_index_kernel<16, int32_t>(char * out, char * inp, int32_t * idx, int64_t idx_stride, int num_ind,
int64_t slice_size_in_bytes, int64_t ind_dim_size, int64_t inp_stride_bytes, int64_t out_stride_bytes);
#endif

#ifdef USE_ROCM
// Batched gathers, used on gfx1250 for gathers that no row gather takes. Blocks
// are one wave (gfx1250 dispatches short-lived multi-wave blocks slowly, and
// single-wave blocks pack best), the grid is capped at one resident set of
// blocks, and each thread takes kBatchedGatherItemsPerThread items per step,
// issuing all of their loads before any store.

// Items per thread step; the per-thread count of index_elementwise_kernel and
// _scatter_gather_elementwise_kernel on ROCm.
constexpr int kBatchedGatherItemsPerThread = 4;

namespace {
// Unsigned word of N bytes, used to copy elements. 16 bytes is a clang vector of
// four 32-bit words, so per-thread arrays of it stay in registers.
using Word16 = uint32_t __attribute__((ext_vector_type(4)));
template <int N>
struct WordOf;
template <>
struct WordOf<1> {
  using type = uint8_t;
};
template <>
struct WordOf<2> {
  using type = uint16_t;
};
template <>
struct WordOf<4> {
  using type = uint32_t;
};
template <>
struct WordOf<8> {
  using type = uint64_t;
};
template <>
struct WordOf<16> {
  using type = Word16;
};

// Arguments of batched_element_gather_kernel, passed as one struct as the
// lambda-based elementwise kernels are: separate pointer arguments cost more
// host time per hipLaunchKernel, which dominates small gathers.
struct ElementGatherParams {
  uint32_t numel;
  OffsetCalculator<3> offset_calc;
  char* out;
  const char* in;
  const char* index;
  int64_t index_size;
  int64_t index_stride_bytes;
  bool allow_neg_indices;
};

// out[i] = in[i] displaced by index[i] * index_stride_bytes, with the offsets of
// out, in and index from offset_calc (tensors 0, 1, 2 of the TensorIterator).
// Each thread issues the index loads of its kBatchedGatherItemsPerThread
// elements, then their element loads, then their stores, so it waits on two
// dependent memory latencies per step rather than two per element.
template <typename element_t, typename index_t>
C10_LAUNCH_BOUNDS_1(C10_WARP_SIZE_UPPER_BOUND)
__global__ void batched_element_gather_kernel(const ElementGatherParams p) {
  constexpr int kWork = kBatchedGatherItemsPerThread;
  const uint32_t step = gridDim.x * blockDim.x * kWork;
  for (uint32_t first = blockIdx.x * blockDim.x * kWork + threadIdx.x; first < p.numel; first += step) {
    char* dst[kWork];
    const char* src[kWork];
#pragma unroll
    for (int i = 0; i < kWork; i++) {
      const uint32_t linear = first + i * blockDim.x;
      if (linear < p.numel) {
        const auto offsets = p.offset_calc.get(linear);
        int64_t ind = *reinterpret_cast<const index_t*>(p.index + offsets[2]);
        if (p.allow_neg_indices && ind < 0) {
          ind += p.index_size;
        }
        CUDA_KERNEL_ASSERT_VERBOSE(ind >= 0 && ind < p.index_size && "element gather kernel index out of bounds", "Expected 0 <= index < index_size(%ld), but got index = %ld", p.index_size, ind);
        dst[i] = p.out + offsets[0];
        src[i] = p.in + offsets[1] + ind * p.index_stride_bytes;
      }
    }
    element_t values[kWork];
#pragma unroll
    for (int i = 0; i < kWork; i++) {
      if (first + i * blockDim.x < p.numel) {
        values[i] = *reinterpret_cast<const element_t*>(src[i]);
      }
    }
#pragma unroll
    for (int i = 0; i < kWork; i++) {
      if (first + i * blockDim.x < p.numel) {
        *reinterpret_cast<element_t*>(dst[i]) = values[i];
      }
    }
  }
}

// Arguments of batched_row_gather_kernel (one struct, as for
// ElementGatherParams).
struct RowGatherParams {
  uint32_t num_chunks;
  at::cuda::detail::IntDivider<uint32_t> chunks_per_row;
  OffsetCalculator<3> row_offset_calc;
  char* out;
  const char* in;
  const char* index;
  int64_t index_size;
  int64_t index_stride_bytes;
  bool allow_neg_indices;
};

// Row gather for an index that is constant along the inner dim (dim 0 of the
// TensorIterator, contiguous in out and in): row r of out is copied from in
// displaced by index[r] * index_stride_bytes, with per-row offsets from
// row_offset_calc over the outer dims. The rows are split into chunks of
// max(kLoadBytes, kStoreBytes); a chunk is read in kLoadBytes words (the
// alignment of the in rows) and written in kStoreBytes words (the alignment of
// the out rows), so misaligned in rows still get wide stores. As in
// batched_element_gather_kernel, each thread issues the loads of all its chunks
// before their stores.
template <int kLoadBytes, int kStoreBytes, typename index_t>
C10_LAUNCH_BOUNDS_1(C10_WARP_SIZE_UPPER_BOUND)
__global__ void batched_row_gather_kernel(const RowGatherParams p) {
  constexpr int kChunkBytes = kLoadBytes > kStoreBytes ? kLoadBytes : kStoreBytes;
  using load_t = typename WordOf<kLoadBytes>::type;
  using store_t = typename WordOf<kStoreBytes>::type;
  union Chunk {
    load_t loads[kChunkBytes / kLoadBytes];
    store_t stores[kChunkBytes / kStoreBytes];
  };
  constexpr int kWork = kBatchedGatherItemsPerThread;
  const uint32_t step = gridDim.x * blockDim.x * kWork;
  for (uint32_t first = blockIdx.x * blockDim.x * kWork + threadIdx.x; first < p.num_chunks; first += step) {
    char* dst[kWork];
    const char* src[kWork];
#pragma unroll
    for (int i = 0; i < kWork; i++) {
      const uint32_t chunk = first + i * blockDim.x;
      if (chunk < p.num_chunks) {
        const auto pos = p.chunks_per_row.divmod(chunk);
        const auto offsets = p.row_offset_calc.get(pos.div);
        int64_t ind = *reinterpret_cast<const index_t*>(p.index + offsets[2]);
        if (p.allow_neg_indices && ind < 0) {
          ind += p.index_size;
        }
        CUDA_KERNEL_ASSERT_VERBOSE(ind >= 0 && ind < p.index_size && "row gather kernel index out of bounds", "Expected 0 <= index < index_size(%ld), but got index = %ld", p.index_size, ind);
        dst[i] = p.out + offsets[0] + pos.mod * kChunkBytes;
        src[i] = p.in + offsets[1] + ind * p.index_stride_bytes + pos.mod * kChunkBytes;
      }
    }
    Chunk chunks[kWork];
#pragma unroll
    for (int i = 0; i < kWork; i++) {
      if (first + i * blockDim.x < p.num_chunks) {
#pragma unroll
        for (int w = 0; w < kChunkBytes / kLoadBytes; w++) {
          chunks[i].loads[w] = reinterpret_cast<const load_t*>(src[i])[w];
        }
      }
    }
#pragma unroll
    for (int i = 0; i < kWork; i++) {
      if (first + i * blockDim.x < p.num_chunks) {
#pragma unroll
        for (int w = 0; w < kChunkBytes / kStoreBytes; w++) {
          reinterpret_cast<store_t*>(dst[i])[w] = chunks[i].stores[w];
        }
      }
    }
  }
}
} // namespace

// One-wave blocks for work_items items of kBatchedGatherItemsPerThread per
// thread step, at most one resident set of them.
static inline std::tuple<dim3, dim3> get_batched_gather_grid(const TensorIteratorBase& iter, int64_t work_items) {
  const auto* props = at::cuda::getDeviceProperties(iter.device().index());
  const int64_t warp_size = props->warpSize;
  const int64_t resident_blocks = static_cast<int64_t>(props->multiProcessorCount) * (props->maxThreadsPerMultiProcessor / warp_size);
  const int64_t blocks = std::min(at::ceil_div(work_items, warp_size * kBatchedGatherItemsPerThread), resident_blocks);
  return {dim3(static_cast<uint32_t>(blocks)), dim3(static_cast<uint32_t>(warp_size))};
}

template <typename index_t>
static void launch_batched_element_gather_kernel_for_index(const TensorIteratorBase& iter, const ElementGatherParams& params) {
  const auto [grid, block] = get_batched_gather_grid(iter, iter.numel());
  const auto stream = at::cuda::getCurrentCUDAStream();
  switch (iter.element_size(0)) {
    case 1: batched_element_gather_kernel<WordOf<1>::type, index_t><<<grid, block, 0, stream>>>(params); break;
    case 2: batched_element_gather_kernel<WordOf<2>::type, index_t><<<grid, block, 0, stream>>>(params); break;
    case 4: batched_element_gather_kernel<WordOf<4>::type, index_t><<<grid, block, 0, stream>>>(params); break;
    case 8: batched_element_gather_kernel<WordOf<8>::type, index_t><<<grid, block, 0, stream>>>(params); break;
    case 16: batched_element_gather_kernel<WordOf<16>::type, index_t><<<grid, block, 0, stream>>>(params); break;
    default: TORCH_INTERNAL_ASSERT(false, "batched element gather: unsupported element size ", iter.element_size(0));
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Load and store word sizes of the batched row gather: the widest words the
// alignments of the in rows (base, outer strides, index stride, row bytes) and
// of the out rows allow. Stores are 16 bytes only for 16-byte aligned out rows;
// otherwise both sides use the narrower word.
static std::pair<int64_t, int64_t> get_batched_row_gather_words(const TensorIteratorBase& iter, int64_t index_stride_bytes) {
  using at::native::memory::get_alignment;
  const auto row_alignment = get_alignment(static_cast<size_t>(iter.shape()[0] * iter.element_size(0)));
  auto store_bytes = std::min(get_alignment(iter.data_ptr(0)), row_alignment);
  auto load_bytes = std::min({get_alignment(iter.data_ptr(1)), row_alignment, get_alignment(static_cast<size_t>(index_stride_bytes))});
  for (int dim = 1; dim < iter.ndim(); dim++) {
    store_bytes = std::min(store_bytes, get_alignment(static_cast<size_t>(iter.strides(0)[dim])));
    load_bytes = std::min(load_bytes, get_alignment(static_cast<size_t>(iter.strides(1)[dim])));
  }
  if (store_bytes < 16) {
    load_bytes = store_bytes = std::min(load_bytes, store_bytes);
  }
  return {static_cast<int64_t>(load_bytes), static_cast<int64_t>(store_bytes)};
}

template <int kLoadBytes, int kStoreBytes, typename index_t>
static void launch_batched_row_gather_kernel_for_words(const TensorIteratorBase& iter, int64_t index_size, int64_t index_stride_bytes, bool allow_neg_indices) {
  constexpr int64_t kChunkBytes = kLoadBytes > kStoreBytes ? kLoadBytes : kStoreBytes;
  const int64_t row_bytes = iter.shape()[0] * iter.element_size(0);
  const int64_t chunks_per_row = row_bytes / kChunkBytes;
  const int64_t num_chunks = iter.numel() / iter.shape()[0] * chunks_per_row;
  // Offsets of each row over the outer dims (1 .. ndim - 1), in bytes.
  const int64_t* outer_strides[3] = {iter.strides(0).data() + 1, iter.strides(1).data() + 1, iter.strides(2).data() + 1};
  const OffsetCalculator<3> row_offset_calc(iter.ndim() - 1, iter.shape().data() + 1, outer_strides);
  const auto [grid, block] = get_batched_gather_grid(iter, num_chunks);
  const RowGatherParams params{
      static_cast<uint32_t>(num_chunks), at::cuda::detail::IntDivider<uint32_t>(static_cast<uint32_t>(chunks_per_row)), row_offset_calc,
      static_cast<char*>(iter.data_ptr(0)), static_cast<const char*>(iter.data_ptr(1)), static_cast<const char*>(iter.data_ptr(2)), index_size,
      index_stride_bytes, allow_neg_indices};
  batched_row_gather_kernel<kLoadBytes, kStoreBytes, index_t><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(params);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename index_t>
static void launch_batched_row_gather_kernel_for_index(const TensorIteratorBase& iter, int64_t index_size, int64_t index_stride_bytes, bool allow_neg_indices) {
  const auto [load_bytes, store_bytes] = get_batched_row_gather_words(iter, index_stride_bytes);
  if (store_bytes == 16) {
    switch (load_bytes) {
      case 16: launch_batched_row_gather_kernel_for_words<16, 16, index_t>(iter, index_size, index_stride_bytes, allow_neg_indices); return;
      case 8: launch_batched_row_gather_kernel_for_words<8, 16, index_t>(iter, index_size, index_stride_bytes, allow_neg_indices); return;
      case 4: launch_batched_row_gather_kernel_for_words<4, 16, index_t>(iter, index_size, index_stride_bytes, allow_neg_indices); return;
      case 2: launch_batched_row_gather_kernel_for_words<2, 16, index_t>(iter, index_size, index_stride_bytes, allow_neg_indices); return;
      default: TORCH_INTERNAL_ASSERT(false, "batched row gather: unsupported load word ", load_bytes);
    }
  }
  switch (store_bytes) {
    case 8: launch_batched_row_gather_kernel_for_words<8, 8, index_t>(iter, index_size, index_stride_bytes, allow_neg_indices); return;
    case 4: launch_batched_row_gather_kernel_for_words<4, 4, index_t>(iter, index_size, index_stride_bytes, allow_neg_indices); return;
    case 2: launch_batched_row_gather_kernel_for_words<2, 2, index_t>(iter, index_size, index_stride_bytes, allow_neg_indices); return;
    default: TORCH_INTERNAL_ASSERT(false, "batched row gather: unsupported word ", store_bytes);
  }
}

bool can_use_batched_element_gather(const TensorIteratorBase& iter) {
  if (iter.ntensors() != 3 || iter.numel() == 0 || !iter.can_use_32bit_indexing() || iter.element_size(0) != iter.element_size(1)) {
    return false;
  }
  const auto element_size = iter.element_size(0);
  const auto index_element_size = iter.element_size(2);
  return (element_size == 1 || element_size == 2 || element_size == 4 || element_size == 8 || element_size == 16) &&
      (index_element_size == sizeof(int64_t) || index_element_size == sizeof(int32_t));
}

void launch_batched_element_gather_kernel(const TensorIteratorBase& iter, int64_t index_size, int64_t index_stride_bytes, bool allow_neg_indices) {
  const ElementGatherParams params{
      static_cast<uint32_t>(iter.numel()), make_offset_calculator<3>(iter), static_cast<char*>(iter.data_ptr(0)),
      static_cast<const char*>(iter.data_ptr(1)), static_cast<const char*>(iter.data_ptr(2)), index_size, index_stride_bytes,
      allow_neg_indices};
  if (iter.element_size(2) == sizeof(int64_t)) {
    launch_batched_element_gather_kernel_for_index<int64_t>(iter, params);
  } else {
    launch_batched_element_gather_kernel_for_index<int32_t>(iter, params);
  }
}

bool can_use_batched_row_gather(const TensorIteratorBase& iter, int64_t index_stride_bytes) {
  if (iter.ntensors() != 3 || iter.ndim() < 2 || iter.numel() == 0 || !iter.can_use_32bit_indexing()) {
    return false;
  }
  const int64_t element_size = iter.element_size(0);
  if (iter.element_size(1) != element_size || iter.strides(2)[0] != 0 || iter.strides(0)[0] != element_size ||
      iter.strides(1)[0] != element_size) {
    return false;
  }
  if (iter.element_size(2) != sizeof(int64_t) && iter.element_size(2) != sizeof(int32_t)) {
    return false;
  }
  const auto [load_bytes, store_bytes] = get_batched_row_gather_words(iter, index_stride_bytes);
  return std::min(load_bytes, store_bytes) >= 2;
}

void launch_batched_row_gather_kernel(const TensorIteratorBase& iter, int64_t index_size, int64_t index_stride_bytes, bool allow_neg_indices) {
  if (iter.element_size(2) == sizeof(int64_t)) {
    launch_batched_row_gather_kernel_for_index<int64_t>(iter, index_size, index_stride_bytes, allow_neg_indices);
  } else {
    launch_batched_row_gather_kernel_for_index<int32_t>(iter, index_size, index_stride_bytes, allow_neg_indices);
  }
}
#endif

// Vectorized scatter: load a vector from src and apply an atomic operation.
template <int Alignment, typename scalar_t>
__device__ __forceinline__ void atomicAddVec(
    scalar_t* dst, const at::native::memory::Vec<Alignment>& vec) {
  constexpr int N = Alignment / sizeof(scalar_t);

  if constexpr (std::is_same_v<scalar_t, float>) {
    #pragma unroll
    for (int i = 0; i < N; i++) {
      atomicAdd(dst + i, vec.f32[i]);
    }
  } else if constexpr (std::is_same_v<scalar_t, double>) {
    const double* vals = reinterpret_cast<const double*>(&vec);
    #pragma unroll
    for (int i = 0; i < N; i++) {
      atomicAdd(dst + i, vals[i]);
    }
  } else if constexpr (std::is_same_v<scalar_t, c10::Half>) {
#if defined(USE_ROCM) || (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 700))
    const c10::Half* vals = reinterpret_cast<const c10::Half*>(&vec);
    #pragma unroll
    for (int i = 0; i < N; i++) {
      gpuAtomicAddNoReturn(dst + i, vals[i]);
    }
#else
    __half2* dst2 = reinterpret_cast<__half2*>(dst);
    const __half2* src2 = reinterpret_cast<const __half2*>(&vec);
    #pragma unroll
    for (int i = 0; i < N / 2; i++) {
      atomicAdd(dst2 + i, src2[i]);
    }
#endif
  } else if constexpr (std::is_same_v<scalar_t, c10::BFloat16>) {
#if defined(USE_ROCM) || (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 800))
    const c10::BFloat16* vals = reinterpret_cast<const c10::BFloat16*>(&vec);
    #pragma unroll
    for (int i = 0; i < N; i++) {
      gpuAtomicAddNoReturn(dst + i, vals[i]);
    }
#else
    __nv_bfloat162* dst2 = reinterpret_cast<__nv_bfloat162*>(dst);
    const __nv_bfloat162* src2 = reinterpret_cast<const __nv_bfloat162*>(&vec);
    #pragma unroll
    for (int i = 0; i < N / 2; i++) {
      atomicAdd(dst2 + i, src2[i]);
    }
#endif
  }
}

#if !defined(USE_ROCM)
template <typename scalar_t>
__device__ __forceinline__ bool packed_isnan(uint16_t bits) {
  if constexpr (std::is_same_v<scalar_t, c10::Half>) {
    return (bits & 0x7c00u) == 0x7c00u && (bits & 0x03ffu) != 0;
  } else {
    return (bits & 0x7f80u) == 0x7f80u && (bits & 0x007fu) != 0;
  }
}

template <typename scalar_t>
__device__ __forceinline__ bool packed_less(uint16_t a, uint16_t b) {
  const uint16_t a_mag = a & 0x7fffu;
  const uint16_t b_mag = b & 0x7fffu;
  if (a_mag == 0 && b_mag == 0) {
    return false;
  }
  const bool a_negative = (a & 0x8000u) != 0;
  const bool b_negative = (b & 0x8000u) != 0;
  if (a_negative != b_negative) {
    return a_negative;
  }
  return a_negative ? a_mag > b_mag : a_mag < b_mag;
}

template <typename scalar_t, bool is_max>
__device__ __forceinline__ uint16_t packed_reduce_bits(
    uint16_t old_bits, uint16_t value_bits) {
  if (packed_isnan<scalar_t>(value_bits)) {
    return value_bits;
  }
  if (packed_isnan<scalar_t>(old_bits)) {
    return old_bits;
  }
  if constexpr (is_max) {
    return packed_less<scalar_t>(old_bits, value_bits) ? value_bits : old_bits;
  } else {
    return packed_less<scalar_t>(value_bits, old_bits) ? value_bits : old_bits;
  }
}

template <typename scalar_t, bool is_max>
__device__ __forceinline__ void atomicReducePackedVec2(
    scalar_t* dst, uint32_t value_bits) {
  auto* address = reinterpret_cast<uint32_t*>(dst);
  uint32_t old = *address;
  uint32_t assumed;
  do {
    assumed = old;
    const uint16_t old_lo = static_cast<uint16_t>(assumed & 0xffffu);
    const uint16_t old_hi = static_cast<uint16_t>(assumed >> 16);
    const uint16_t value_lo = static_cast<uint16_t>(value_bits & 0xffffu);
    const uint16_t value_hi = static_cast<uint16_t>(value_bits >> 16);
    const uint32_t result =
        static_cast<uint32_t>(packed_reduce_bits<scalar_t, is_max>(old_lo, value_lo)) |
        (static_cast<uint32_t>(packed_reduce_bits<scalar_t, is_max>(old_hi, value_hi)) << 16);
    if (result == assumed) {
      return;
    }
    old = atomicCAS(address, assumed, result);
  } while (old != assumed);
}

template <int Alignment, typename scalar_t>
__device__ __forceinline__ void atomicMinVec(
    scalar_t* dst, const at::native::memory::Vec<Alignment>& vec) {
  constexpr int N = Alignment / sizeof(scalar_t);
  static_assert(std::is_same_v<scalar_t, c10::Half> ||
                std::is_same_v<scalar_t, c10::BFloat16>);
  static_assert(N % 2 == 0, "packed CAS requires pairs of 16-bit values");
  const uint32_t* values = reinterpret_cast<const uint32_t*>(&vec);
  #pragma unroll
  for (int i = 0; i < N / 2; ++i) {
    atomicReducePackedVec2<scalar_t, false>(dst + 2 * i, values[i]);
  }
}

template <int Alignment, typename scalar_t>
__device__ __forceinline__ void atomicMaxVec(
    scalar_t* dst, const at::native::memory::Vec<Alignment>& vec) {
  constexpr int N = Alignment / sizeof(scalar_t);
  static_assert(std::is_same_v<scalar_t, c10::Half> ||
                std::is_same_v<scalar_t, c10::BFloat16>);
  static_assert(N % 2 == 0, "packed CAS requires pairs of 16-bit values");
  const uint32_t* values = reinterpret_cast<const uint32_t*>(&vec);
  #pragma unroll
  for (int i = 0; i < N / 2; ++i) {
    atomicReducePackedVec2<scalar_t, true>(dst + 2 * i, values[i]);
  }
}

struct ScatterMinOp {
  template <int Alignment, typename scalar_t>
  __device__ __forceinline__ static void apply(
      scalar_t* dst, const at::native::memory::Vec<Alignment>& vec) {
    atomicMinVec<Alignment>(dst, vec);
  }
};

struct ScatterMaxOp {
  template <int Alignment, typename scalar_t>
  __device__ __forceinline__ static void apply(
      scalar_t* dst, const at::native::memory::Vec<Alignment>& vec) {
    atomicMaxVec<Alignment>(dst, vec);
  }
};
#endif

struct ScatterAddOp {
  template <int Alignment, typename scalar_t>
  __device__ __forceinline__ static void apply(
      scalar_t* dst, const at::native::memory::Vec<Alignment>& vec) {
    atomicAddVec<Alignment>(dst, vec);
  }
};

struct ScatterCopyOp {
  static constexpr int threads_per_entry = 32;
  static constexpr int entries_per_block = 8;

  template <int Alignment, typename scalar_t>
  __device__ __forceinline__ static void apply(
      scalar_t* dst, const at::native::memory::Vec<Alignment>& vec) {
    at::native::memory::st_vec<Alignment>(reinterpret_cast<char*>(dst), vec);
  }
};

template <typename reduce_op, int Alignment, typename scalar_t, typename index_t>
__global__ void vectorized_scatter_kernel(
    scalar_t* __restrict__ self_data,
    const scalar_t* __restrict__ src_data,
    const index_t* __restrict__ idx,
    int num_ind,
    int64_t slice_size_bytes,
    int64_t self_dim_size,
    int64_t self_stride_bytes,
    int64_t src_stride_bytes,
    int threads_per_entry,
    int entries_per_block) {

  if constexpr (std::is_same_v<reduce_op, ScatterCopyOp>) {
    threads_per_entry = ScatterCopyOp::threads_per_entry;
    entries_per_block = ScatterCopyOp::entries_per_block;
  }
  int entry_in_block = threadIdx.x / threads_per_entry;
  int lane = threadIdx.x - entry_in_block * threads_per_entry;
  for (int entry_id = blockIdx.x * entries_per_block + entry_in_block;
       entry_id < num_ind;
       entry_id += gridDim.x * entries_per_block) {
    int64_t ind = idx[entry_id];
    CUDA_KERNEL_ASSERT_VERBOSE(ind >= 0 && ind < self_dim_size && "vectorized scatter kernel index out of bounds",
        "Expected 0 <= index < self_dim_size(%ld), but got index = %ld", self_dim_size, ind);

    scalar_t* dst_slice = reinterpret_cast<scalar_t*>(
        reinterpret_cast<char*>(self_data) + ind * self_stride_bytes);

    for (uint32_t off = (blockIdx.y * threads_per_entry + lane) * Alignment;
         off < slice_size_bytes;
         off += gridDim.y * threads_per_entry * Alignment) {
      auto vec = at::native::memory::ld_vec<Alignment>(
          reinterpret_cast<const char*>(src_data) + entry_id * static_cast<int32_t>(src_stride_bytes) + off);

      scalar_t* dst = reinterpret_cast<scalar_t*>(
          reinterpret_cast<char*>(dst_slice) + off);

      reduce_op::template apply<Alignment>(dst, vec);
    }
    if constexpr (!std::is_same_v<reduce_op, ScatterCopyOp>) return;
  }
}

template <typename atomic_op, int64_t Alignment, typename scalar_t, typename index_t>
void vectorized_scatter_kernel_launch(
    scalar_t* self_data, const scalar_t* src_data, index_t* idx, int num_ind,
    int64_t slice_size_in_bytes, int64_t self_dim_size,
    int64_t self_stride_bytes, int64_t src_stride_bytes) {

  constexpr int64_t max_num_threads = 256;
  int64_t num_vectors = slice_size_in_bytes / Alignment;
  int64_t threads_per_entry = std::min(
      at::round_up(num_vectors, static_cast<int64_t>(at::cuda::warp_size())),
      max_num_threads);
  int64_t entries_per_block = max_num_threads / threads_per_entry;
  if constexpr (std::is_same_v<atomic_op, ScatterCopyOp>) {
    threads_per_entry = ScatterCopyOp::threads_per_entry;
    entries_per_block = ScatterCopyOp::entries_per_block;
  }
  int64_t block_size = entries_per_block * threads_per_entry;
  int64_t grid_x = at::ceil_div(static_cast<int64_t>(num_ind), entries_per_block);
  uint32_t grid_y = at::cuda::getCurrentDeviceProperties()->maxGridSize[1];
  grid_y = std::min(
      static_cast<uint32_t>(at::ceil_div(num_vectors, threads_per_entry)),
      grid_y);
  if constexpr (std::is_same_v<atomic_op, ScatterCopyOp>) {
    const int64_t max_blocks =
        at::cuda::getCurrentDeviceProperties()->multiProcessorCount * 8;
    grid_x = std::min(grid_x, max_blocks);
    // Split wide slices when there are too few rows to fill the grid.
    grid_y = std::min(static_cast<int64_t>(grid_y), max_blocks / grid_x);
  }

  dim3 grid = {static_cast<uint32_t>(grid_x), grid_y, 1};
  vectorized_scatter_kernel<atomic_op, Alignment, scalar_t, index_t>
      <<<grid, block_size, 0, at::cuda::getCurrentCUDAStream()>>>(
      self_data, src_data, idx, num_ind, slice_size_in_bytes,
      self_dim_size, self_stride_bytes, src_stride_bytes,
      static_cast<int>(threads_per_entry), static_cast<int>(entries_per_block));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

#define INSTANTIATE_SCATTER_OP(op, scalar_t) \
template void vectorized_scatter_kernel_launch<op, 16, scalar_t, int64_t>( \
    scalar_t*, const scalar_t*, int64_t*, int, int64_t, int64_t, int64_t, int64_t); \
template void vectorized_scatter_kernel_launch<op, 16, scalar_t, int32_t>( \
    scalar_t*, const scalar_t*, int32_t*, int, int64_t, int64_t, int64_t, int64_t);

#define INSTANTIATE_SCATTER_ADD(scalar_t) INSTANTIATE_SCATTER_OP(ScatterAddOp, scalar_t)
INSTANTIATE_SCATTER_OP(ScatterCopyOp, char)
#if !defined(USE_ROCM)
#define INSTANTIATE_SCATTER_MINMAX(scalar_t) \
  INSTANTIATE_SCATTER_OP(ScatterMinOp, scalar_t) \
  INSTANTIATE_SCATTER_OP(ScatterMaxOp, scalar_t)

INSTANTIATE_SCATTER_ADD(float)
INSTANTIATE_SCATTER_ADD(double)
INSTANTIATE_SCATTER_ADD(c10::Half)
INSTANTIATE_SCATTER_ADD(c10::BFloat16)
INSTANTIATE_SCATTER_MINMAX(c10::Half)
INSTANTIATE_SCATTER_MINMAX(c10::BFloat16)
#undef INSTANTIATE_SCATTER_MINMAX
#endif
#undef INSTANTIATE_SCATTER_ADD
#undef INSTANTIATE_SCATTER_OP


}
