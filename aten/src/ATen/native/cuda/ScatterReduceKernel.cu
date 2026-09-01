// Scatter-reduce min/max fast paths: cp.async plus packed CAS on SM80-SM89, red.* on SM90+.

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Atomic.cuh>
#include <ATen/ceil_div.h>
#include <c10/macros/Macros.h>
#include <c10/util/BFloat16.h>
#include <c10/util/Exception.h>
#include <c10/util/Half.h>

#include <algorithm>
#include <cstdint>
#include <type_traits>

namespace at::native {

#if !defined(USE_ROCM) && defined(CUDA_VERSION) && CUDA_VERSION >= 12080 && \
    defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
namespace scatter_reduce {

template <int bytes>
__device__ __forceinline__ void async_copy(void* smem_dst, const void* global_src) {
  uint64_t smem_addr;
  asm volatile(
      "cvta.to.shared::cta.u64 %0, %1;"
      : "=l"(smem_addr)
      : "l"(reinterpret_cast<uint64_t>(smem_dst)));
  asm volatile(
      "cp.async.ca.shared.global [%0], [%1], %2;"
      : : "l"(smem_addr), "l"(global_src), "n"(bytes) : "memory");
}

__device__ __forceinline__ void async_commit_group() {
  asm volatile("cp.async.commit_group;" ::: "memory");
}

__device__ __forceinline__ void async_wait_group_0() {
  asm volatile("cp.async.wait_group 0;" ::: "memory");
}

#define DEFINE_VECTOR_RED16(name, op, type)                                  \
  template <int vec_size>                                                     \
  __device__ __forceinline__ void name(void* dst, const uint32_t* src) {     \
    if constexpr (vec_size == 8) {                                           \
      uint32_t x0 = src[0], x1 = src[1], x2 = src[2], x3 = src[3];          \
      asm volatile(                                                          \
          "{ .reg .b16 h<9>;\n"                                             \
          "mov.b32 {h1, h2}, %1;\n"                                        \
          "mov.b32 {h3, h4}, %2;\n"                                        \
          "mov.b32 {h5, h6}, %3;\n"                                        \
          "mov.b32 {h7, h8}, %4;\n"                                        \
          "red.relaxed.gpu.global." op ".noftz.v8." type                   \
          " [%0], {h1, h2, h3, h4, h5, h6, h7, h8};\n"                      \
          "}"                                                               \
          : : "l"(dst), "r"(x0), "r"(x1), "r"(x2), "r"(x3) : "memory"); \
    } else if constexpr (vec_size == 4) {                                    \
      uint32_t x0 = src[0], x1 = src[1];                                    \
      asm volatile(                                                          \
          "{ .reg .b16 h<5>;\n"                                             \
          "mov.b32 {h1, h2}, %1;\n"                                        \
          "mov.b32 {h3, h4}, %2;\n"                                        \
          "red.relaxed.gpu.global." op ".noftz.v4." type                   \
          " [%0], {h1, h2, h3, h4};\n"                                     \
          "}"                                                               \
          : : "l"(dst), "r"(x0), "r"(x1) : "memory");                    \
    } else {                                                                 \
      static_assert(vec_size == 2,                                           \
                    "only v8, v4, and v2 reductions are supported");       \
      uint32_t x0 = src[0];                                                  \
      asm volatile(                                                          \
          "{ .reg .b16 h<3>;\n"                                             \
          "mov.b32 {h1, h2}, %1;\n"                                        \
          "red.relaxed.gpu.global." op ".noftz.v2." type                   \
          " [%0], {h1, h2};\n"                                             \
          "}"                                                               \
          : : "l"(dst), "r"(x0) : "memory");                              \
    }                                                                        \
  }

DEFINE_VECTOR_RED16(red_f16_max, "max", "f16")
DEFINE_VECTOR_RED16(red_f16_min, "min", "f16")
DEFINE_VECTOR_RED16(red_bf16_max, "max", "bf16")
DEFINE_VECTOR_RED16(red_bf16_min, "min", "bf16")
#undef DEFINE_VECTOR_RED16

} // namespace scatter_reduce
#endif

template <typename scalar_t, typename index_t, bool is_max, int vec_size>
__global__ void vectorized_scatter_reduce_minmax_kernel(
    scalar_t* __restrict__ self_data,
    const scalar_t* __restrict__ src_data,
    const index_t* __restrict__ idx,
    int num_ind,
    int D,
    int64_t self_dim_size,
    int64_t self_stride,
    int64_t src_stride,
    int entries_per_block,
    int tile_elems) {
#if !defined(USE_ROCM) && defined(CUDA_VERSION) && CUDA_VERSION >= 12080 && \
    defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  extern __shared__ char smem_raw[];
  constexpr int threads_per_entry = C10_WARP_SIZE;
  constexpr int copy_bytes = vec_size * sizeof(scalar_t);
  int entry_in_block = threadIdx.x / threads_per_entry;
  int lane = threadIdx.x - entry_in_block * threads_per_entry;
  int buf_elems = 2 * tile_elems;
  scalar_t* buf0 = reinterpret_cast<scalar_t*>(smem_raw) + entry_in_block * buf_elems;

  int entry_id = blockIdx.x * entries_per_block + entry_in_block;
  if (entry_id >= num_ind) {
    return;
  }
  int64_t ind = idx[entry_id];
  CUDA_KERNEL_ASSERT_VERBOSE(
      ind >= 0 && ind < self_dim_size && "vectorized scatter min/max index out of bounds",
      "Expected 0 <= index < self_dim_size(%ld), but got index = %ld",
      self_dim_size,
      ind);
  const scalar_t* src_entry = src_data + static_cast<int64_t>(entry_id) * src_stride;
  scalar_t* dst_entry = self_data + ind * self_stride;

  int off = blockIdx.y * tile_elems;
  int phase = 0;
  if (off < D) {
    int vector_off = off + lane * vec_size;
    if (vector_off < D) {
      scatter_reduce::async_copy<copy_bytes>(
          buf0 + lane * vec_size, src_entry + vector_off);
    }
    scatter_reduce::async_commit_group();
  }
  for (; off < D; off += gridDim.y * tile_elems, ++phase) {
    int cur = phase & 1;
    scatter_reduce::async_wait_group_0();
    __syncwarp();

    int next_off = off + gridDim.y * tile_elems;
    if (next_off < D) {
      int next_vector_off = next_off + lane * vec_size;
      if (next_vector_off < D) {
        scatter_reduce::async_copy<copy_bytes>(
            buf0 + (cur ^ 1) * tile_elems + lane * vec_size,
            src_entry + next_vector_off);
      }
      scatter_reduce::async_commit_group();
    }

    int vector_off = off + lane * vec_size;
    if (vector_off < D) {
      const uint32_t* src_vector = reinterpret_cast<const uint32_t*>(
          buf0 + cur * tile_elems + lane * vec_size);
      if constexpr (std::is_same_v<scalar_t, c10::Half>) {
        if constexpr (is_max) {
          scatter_reduce::red_f16_max<vec_size>(dst_entry + vector_off, src_vector);
        } else {
          scatter_reduce::red_f16_min<vec_size>(dst_entry + vector_off, src_vector);
        }
      } else {
        if constexpr (is_max) {
          scatter_reduce::red_bf16_max<vec_size>(dst_entry + vector_off, src_vector);
        } else {
          scatter_reduce::red_bf16_min<vec_size>(dst_entry + vector_off, src_vector);
        }
      }
    }
  }
#else
  CUDA_KERNEL_ASSERT(
      false && "vectorized_scatter_reduce_minmax_kernel requires sm_90+");
#endif
}

template <typename scalar_t, typename index_t, bool is_max, int vec_size>
void vectorized_scatter_reduce_minmax_kernel_launch(
    scalar_t* self_data,
    const scalar_t* src_data,
    index_t* idx,
    int num_ind,
    int D,
    int64_t self_dim_size,
    int64_t self_stride_bytes,
    int64_t src_stride_bytes) {
#if !defined(USE_ROCM) && defined(CUDA_VERSION) && CUDA_VERSION >= 12080
  constexpr int max_threads = 256;
  constexpr int threads_per_entry = C10_WARP_SIZE;
  constexpr int entries_per_block = max_threads / threads_per_entry;
  constexpr int tile_elems = C10_WARP_SIZE * vec_size;
  constexpr int min_tiles_per_block = 4;
  int num_tiles = at::ceil_div(D, tile_elems);
  int grid_x = at::ceil_div(num_ind, entries_per_block);
  uint32_t grid_y = std::min(
      static_cast<uint32_t>(std::max(1, num_tiles / min_tiles_per_block)),
      static_cast<uint32_t>(at::cuda::getCurrentDeviceProperties()->maxGridSize[1]));
  int64_t self_stride = self_stride_bytes / sizeof(scalar_t);
  int64_t src_stride = src_stride_bytes / sizeof(scalar_t);
  int smem = entries_per_block * 2 * tile_elems * static_cast<int>(sizeof(scalar_t));
  dim3 grid = {static_cast<uint32_t>(grid_x), grid_y, 1};

  vectorized_scatter_reduce_minmax_kernel<scalar_t, index_t, is_max, vec_size>
      <<<grid, max_threads, smem, at::cuda::getCurrentCUDAStream()>>>(
          self_data,
          src_data,
          idx,
          num_ind,
          D,
          self_dim_size,
          self_stride,
          src_stride,
          entries_per_block,
          tile_elems);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
#else
  TORCH_CHECK(false, "vectorized scatter min/max requires CUDA 12.8+ and NVIDIA GPU");
#endif
}

#define INSTANTIATE_VECTOR_SCATTER_REDUCE_MINMAX(scalar_t, index_t, is_max) \
  template void vectorized_scatter_reduce_minmax_kernel_launch<             \
      scalar_t, index_t, is_max, 8>(                                         \
      scalar_t*, const scalar_t*, index_t*, int, int, int64_t, int64_t, int64_t); \
  template void vectorized_scatter_reduce_minmax_kernel_launch<             \
      scalar_t, index_t, is_max, 4>(                                         \
      scalar_t*, const scalar_t*, index_t*, int, int, int64_t, int64_t, int64_t); \
  template void vectorized_scatter_reduce_minmax_kernel_launch<             \
      scalar_t, index_t, is_max, 2>(                                         \
      scalar_t*, const scalar_t*, index_t*, int, int, int64_t, int64_t, int64_t);

#define INSTANTIATE_VECTOR_SCATTER_REDUCE_MINMAX_INDEX(scalar_t, index_t) \
  INSTANTIATE_VECTOR_SCATTER_REDUCE_MINMAX(scalar_t, index_t, true)       \
  INSTANTIATE_VECTOR_SCATTER_REDUCE_MINMAX(scalar_t, index_t, false)

INSTANTIATE_VECTOR_SCATTER_REDUCE_MINMAX_INDEX(c10::Half, int64_t)
INSTANTIATE_VECTOR_SCATTER_REDUCE_MINMAX_INDEX(c10::Half, int32_t)
INSTANTIATE_VECTOR_SCATTER_REDUCE_MINMAX_INDEX(c10::BFloat16, int64_t)
INSTANTIATE_VECTOR_SCATTER_REDUCE_MINMAX_INDEX(c10::BFloat16, int32_t)
#undef INSTANTIATE_VECTOR_SCATTER_REDUCE_MINMAX_INDEX
#undef INSTANTIATE_VECTOR_SCATTER_REDUCE_MINMAX

#if !defined(USE_ROCM) && defined(CUDA_VERSION) && CUDA_VERSION >= 11000 && \
    defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800 && __CUDA_ARCH__ < 900
namespace ampere_scatter_reduce {

template <int bytes>
__device__ __forceinline__ void async_copy(void* smem_dst, const void* global_src) {
  uint64_t smem_addr;
  asm volatile(
      "cvta.to.shared::cta.u64 %0, %1;"
      : "=l"(smem_addr)
      : "l"(reinterpret_cast<uint64_t>(smem_dst)));
  asm volatile(
      "cp.async.ca.shared.global [%0], [%1], %2;"
      : : "l"(smem_addr), "l"(global_src), "n"(bytes) : "memory");
}

__device__ __forceinline__ void commit_group() {
  asm volatile("cp.async.commit_group;" ::: "memory");
}

__device__ __forceinline__ void wait_group_0() {
  asm volatile("cp.async.wait_group 0;" ::: "memory");
}

__device__ __forceinline__ bool half_isnan(uint16_t bits) {
  return (bits & 0x7c00u) == 0x7c00u && (bits & 0x03ffu) != 0;
}

__device__ __forceinline__ bool half_less(uint16_t a, uint16_t b) {
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

template <bool is_max>
__device__ __forceinline__ uint16_t half_reduce_bits(
    uint16_t old_bits, uint16_t value_bits) {
  // Match torch::safe_max/safe_min: NaN in the incoming value wins, while an
  // existing NaN is retained. Equal values retain the existing bit pattern.
  if (half_isnan(value_bits)) {
    return value_bits;
  }
  if (half_isnan(old_bits)) {
    return old_bits;
  }
  if constexpr (is_max) {
    return half_less(old_bits, value_bits) ? value_bits : old_bits;
  } else {
    return half_less(value_bits, old_bits) ? value_bits : old_bits;
  }
}

template <bool is_max>
__device__ __forceinline__ void atomic_reduce_half2(
    c10::Half* dst, uint32_t value_bits) {
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
        static_cast<uint32_t>(half_reduce_bits<is_max>(old_lo, value_lo)) |
        (static_cast<uint32_t>(half_reduce_bits<is_max>(old_hi, value_hi)) << 16);
    if (result == assumed) {
      return;
    }
    old = atomicCAS(address, assumed, result);
  } while (old != assumed);
}

__device__ __forceinline__ bool bfloat16_isnan(uint16_t bits) {
  return (bits & 0x7f80u) == 0x7f80u && (bits & 0x007fu) != 0;
}

__device__ __forceinline__ bool bfloat16_less(uint16_t a, uint16_t b) {
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

template <bool is_max>
__device__ __forceinline__ uint16_t bfloat16_reduce_bits(
    uint16_t old_bits, uint16_t value_bits) {
  if (bfloat16_isnan(value_bits)) {
    return value_bits;
  }
  if (bfloat16_isnan(old_bits)) {
    return old_bits;
  }
  if constexpr (is_max) {
    return bfloat16_less(old_bits, value_bits) ? value_bits : old_bits;
  } else {
    return bfloat16_less(value_bits, old_bits) ? value_bits : old_bits;
  }
}

template <bool is_max>
__device__ __forceinline__ void atomic_reduce_bfloat162(
    c10::BFloat16* dst, uint32_t value_bits) {
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
        static_cast<uint32_t>(bfloat16_reduce_bits<is_max>(old_lo, value_lo)) |
        (static_cast<uint32_t>(bfloat16_reduce_bits<is_max>(old_hi, value_hi)) << 16);
    if (result == assumed) {
      return;
    }
    old = atomicCAS(address, assumed, result);
  } while (old != assumed);
}

} // namespace ampere_scatter_reduce
#endif

template <typename scalar_t, typename index_t, bool is_max>
__global__ void ampere_scatter_reduce_minmax_kernel(
    scalar_t* __restrict__ self_data,
    const scalar_t* __restrict__ src_data,
    const index_t* __restrict__ idx,
    int num_ind,
    int D,
    int64_t self_dim_size,
    int64_t self_stride,
    int64_t src_stride,
    int entries_per_block,
    int tile_elems) {
#if !defined(USE_ROCM) && defined(CUDA_VERSION) && CUDA_VERSION >= 11000 && \
    defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800 && __CUDA_ARCH__ < 900
  extern __shared__ char smem_raw[];
  constexpr int threads_per_entry = C10_WARP_SIZE;
  constexpr int vec_size = 16 / sizeof(scalar_t);
  constexpr int copy_bytes = 16;
  const int entry_in_block = threadIdx.x / threads_per_entry;
  const int lane = threadIdx.x - entry_in_block * threads_per_entry;
  const int entry_id = blockIdx.x * entries_per_block + entry_in_block;
  if (entry_id >= num_ind) {
    return;
  }

  const int64_t ind = idx[entry_id];
  CUDA_KERNEL_ASSERT_VERBOSE(
      ind >= 0 && ind < self_dim_size && "ampere scatter min/max index out of bounds",
      "Expected 0 <= index < self_dim_size(%ld), but got index = %ld",
      self_dim_size,
      ind);
  const scalar_t* src_entry = src_data + static_cast<int64_t>(entry_id) * src_stride;
  scalar_t* dst_entry = self_data + ind * self_stride;
  scalar_t* buffer = reinterpret_cast<scalar_t*>(smem_raw) +
      entry_in_block * 2 * tile_elems;

  int off = blockIdx.y * tile_elems;
  int phase = 0;
  if (off < D) {
    const int vector_off = off + lane * vec_size;
    if (vector_off < D) {
      ampere_scatter_reduce::async_copy<copy_bytes>(
          buffer + lane * vec_size, src_entry + vector_off);
    }
    ampere_scatter_reduce::commit_group();
  }

  for (; off < D; off += gridDim.y * tile_elems, ++phase) {
    const int cur = phase & 1;
    ampere_scatter_reduce::wait_group_0();
    __syncwarp();

    const int next_off = off + gridDim.y * tile_elems;
    if (next_off < D) {
      const int next_vector_off = next_off + lane * vec_size;
      if (next_vector_off < D) {
        ampere_scatter_reduce::async_copy<copy_bytes>(
            buffer + (cur ^ 1) * tile_elems + lane * vec_size,
            src_entry + next_vector_off);
      }
      ampere_scatter_reduce::commit_group();
    }

    const int vector_off = off + lane * vec_size;
    if (vector_off < D) {
      const int count = min(vec_size, D - vector_off);
      scalar_t* values = buffer + cur * tile_elems + lane * vec_size;
      if constexpr (std::is_same_v<scalar_t, c10::Half>) {
        const uint32_t* packed_values =
            reinterpret_cast<const uint32_t*>(values);
#pragma unroll
        for (int j = 0; j < vec_size / 2; ++j) {
          if (2 * j < count) {
            ampere_scatter_reduce::atomic_reduce_half2<is_max>(
                dst_entry + vector_off + 2 * j, packed_values[j]);
          }
        }
      } else if constexpr (std::is_same_v<scalar_t, c10::BFloat16>) {
        const uint32_t* packed_values =
            reinterpret_cast<const uint32_t*>(values);
#pragma unroll
        for (int j = 0; j < vec_size / 2; ++j) {
          if (2 * j < count) {
            ampere_scatter_reduce::atomic_reduce_bfloat162<is_max>(
                dst_entry + vector_off + 2 * j, packed_values[j]);
          }
        }
      }
    }
  }
#else
  CUDA_KERNEL_ASSERT(false && "Ampere scatter min/max requires sm_80+");
#endif
}

template <typename scalar_t, typename index_t, bool is_max>
void ampere_scatter_reduce_minmax_kernel_launch(
    scalar_t* self_data,
    const scalar_t* src_data,
    index_t* idx,
    int num_ind,
    int D,
    int64_t self_dim_size,
    int64_t self_stride_bytes,
    int64_t src_stride_bytes) {
#if !defined(USE_ROCM) && defined(CUDA_VERSION) && CUDA_VERSION >= 11000
  constexpr int max_threads = 256;
  constexpr int threads_per_entry = C10_WARP_SIZE;
  constexpr int entries_per_block = max_threads / threads_per_entry;
  constexpr int tile_elems = C10_WARP_SIZE * (16 / sizeof(scalar_t));
  constexpr int min_tiles_per_block = 4;
  const int num_tiles = at::ceil_div(D, tile_elems);
  const int grid_x = at::ceil_div(num_ind, entries_per_block);
  const uint32_t grid_y = std::min(
      static_cast<uint32_t>(std::max(1, num_tiles / min_tiles_per_block)),
      static_cast<uint32_t>(at::cuda::getCurrentDeviceProperties()->maxGridSize[1]));
  const int64_t self_stride = self_stride_bytes / sizeof(scalar_t);
  const int64_t src_stride = src_stride_bytes / sizeof(scalar_t);
  const int smem = entries_per_block * 2 * tile_elems * static_cast<int>(sizeof(scalar_t));
  dim3 grid = {static_cast<uint32_t>(grid_x), grid_y, 1};

  ampere_scatter_reduce_minmax_kernel<scalar_t, index_t, is_max>
      <<<grid, max_threads, smem, at::cuda::getCurrentCUDAStream()>>>(
          self_data,
          src_data,
          idx,
          num_ind,
          D,
          self_dim_size,
          self_stride,
          src_stride,
          entries_per_block,
          tile_elems);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
#else
  TORCH_CHECK(false, "Ampere scatter min/max requires CUDA 11+ and NVIDIA GPU");
#endif
}

#define INSTANTIATE_AMPERE_SCATTER_REDUCE_MINMAX(scalar_t, index_t, is_max) \
  template void ampere_scatter_reduce_minmax_kernel_launch<                 \
      scalar_t, index_t, is_max>(                                             \
      scalar_t*, const scalar_t*, index_t*, int, int, int64_t, int64_t, int64_t);

#define INSTANTIATE_AMPERE_SCATTER_REDUCE_MINMAX_INDEX(scalar_t, index_t) \
  INSTANTIATE_AMPERE_SCATTER_REDUCE_MINMAX(scalar_t, index_t, true)       \
  INSTANTIATE_AMPERE_SCATTER_REDUCE_MINMAX(scalar_t, index_t, false)

INSTANTIATE_AMPERE_SCATTER_REDUCE_MINMAX_INDEX(c10::Half, int64_t)
INSTANTIATE_AMPERE_SCATTER_REDUCE_MINMAX_INDEX(c10::Half, int32_t)
INSTANTIATE_AMPERE_SCATTER_REDUCE_MINMAX_INDEX(c10::BFloat16, int64_t)
INSTANTIATE_AMPERE_SCATTER_REDUCE_MINMAX_INDEX(c10::BFloat16, int32_t)
#undef INSTANTIATE_AMPERE_SCATTER_REDUCE_MINMAX_INDEX
#undef INSTANTIATE_AMPERE_SCATTER_REDUCE_MINMAX

} // namespace at::native
