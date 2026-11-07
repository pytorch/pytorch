// File: gpu_opt_elementwise_kernels/dispatch.cuh
// GPU elementwise dispatch entry points.
// Per-ndim/arity predicates live in the corresponding subdirectories.

#pragma once

#include "utils/env.cuh"
#include "nocast_elementwise/include.cuh"
#include "cast_elementwise/include.cuh"
#include "cast_contiguous_unrolled/include.cuh"

// ── Dispatch table macros ────────────────────────────────────
// X-macro tables eliminate the manual if/else chain in each entry
// function.  The helper expands to a single arity+ndim check.
#define GPU_DISPATCH_ONE(PREFIX, NDIM, ARITY)                    \
  if constexpr (traits::arity == ARITY) {                        \
    if (iter.ndim() == NDIM) {                                   \
      return PREFIX##_##NDIM##_##ARITY(iter, f, data);           \
    }                                                            \
  }

// nocast_elementwise table (arity 1..3, ndim 1..5).
#define GPU_NOCAST_DISPATCH_TABLE(F)                             \
  F(try_launch_gpu, 1, 1) F(try_launch_gpu, 2, 1)                \
  F(try_launch_gpu, 3, 1) F(try_launch_gpu, 4, 1)                \
  F(try_launch_gpu, 5, 1) F(try_launch_gpu, 1, 2)                \
  F(try_launch_gpu, 2, 2) F(try_launch_gpu, 3, 2)                \
  F(try_launch_gpu, 4, 2) F(try_launch_gpu, 1, 3)

// cast_elementwise table (arity 1..2, ndim 1..3).
#define GPU_CAST_DISPATCH_TABLE(F)                               \
  F(try_launch_gpu_cast_elementwise, 2, 1)                        \
  F(try_launch_gpu_cast_elementwise, 1, 2)                        \
  F(try_launch_gpu_cast_elementwise, 2, 2)                        \
  F(try_launch_gpu_cast_elementwise, 3, 2)

// cast_contiguous_unrolled table (arity 1..2, ndim 1).
#define GPU_CAST_CONTIG_DISPATCH_TABLE(F)                        \
  F(try_launch_gpu_cast_contiguous, 1, 1)                         \
  F(try_launch_gpu_cast_contiguous, 1, 2)

struct GPUPackedLayout {
  c10::IntDivider<uint32_t> sizes[2];
  int64_t strides[3][3];
};

template <typename T, int vt>
__device__ memory::aligned_vector<T, vt> gpu_packed_load(
    const char* data, int64_t offset, int64_t stride) {
  using vec_t = memory::aligned_vector<T, vt>;
  if constexpr (!std::is_same_v<T, bool>) {
    if (stride == static_cast<int64_t>(sizeof(T))) {
      return *reinterpret_cast<const vec_t*>(data + offset);
    }
  }
  vec_t result;
  if (stride == 0) {
    const T value = c10::load(reinterpret_cast<const T*>(data + offset));
#pragma unroll
    for (int i = 0; i < vt; ++i) result.val[i] = value;
  } else {
#pragma unroll
    for (int i = 0; i < vt; ++i) {
      result.val[i] = c10::load(reinterpret_cast<const T*>(data + offset + i * stride));
    }
  }
  return result;
}

template <int vt, typename storage0_t, typename storage1_t, typename func_t>
C10_LAUNCH_BOUNDS_1(256)
__global__ void elementwise_kernel_binary_packed(
    uint32_t N, char* out, const char* a, const char* b,
    GPUPackedLayout layout, func_t f) {
  using traits = function_traits<func_t>;
  using res_t = typename traits::result_type;
  const uint32_t index = (blockIdx.x * blockDim.x + threadIdx.x) * vt;
  if (index >= N) return;
  const auto x = layout.sizes[0].divmod(index);
  const auto y = layout.sizes[1].divmod(x.div);
  int64_t offsets[3];
#pragma unroll
  for (int i = 0; i < 3; ++i) {
    offsets[i] = x.mod * layout.strides[i][0] +
        y.mod * layout.strides[i][1] + y.div * layout.strides[i][2];
  }
  const auto av = gpu_packed_load<storage0_t, vt>(a, offsets[1], layout.strides[1][0]);
  const auto bv = gpu_packed_load<storage1_t, vt>(b, offsets[2], layout.strides[2][0]);
  memory::aligned_vector<res_t, vt> result;
#pragma unroll
  for (int i = 0; i < vt; ++i) result.val[i] = f(av.val[i], bv.val[i]);
  *reinterpret_cast<memory::aligned_vector<res_t, vt>*>(out + offsets[0]) = result;
}

template <typename storage0_t, typename storage1_t, typename func_t, typename array_t>
static inline bool try_launch_gpu_binary_packed(
    TensorIteratorBase& iter, const func_t& f, array_t data) {
  using traits = function_traits<func_t>;
  using res_t = typename traits::result_type;
  static_assert(traits::arity == 2);
  constexpr int pack = sizeof(res_t) == 8 ? 2 : 4;
  if ((iter.ndim() != 2 && iter.ndim() != 3) || iter.ntensors() != 3 ||
      iter.noutputs() != 1 || iter.numel() <= 0 || !iter.can_use_32bit_indexing() ||
      iter.strides(0)[0] != static_cast<int64_t>(sizeof(res_t))) return false;
  GPUPackedLayout layout{};
  for (int d = 0; d < 2; ++d) {
    layout.sizes[d] = c10::IntDivider<uint32_t>(static_cast<uint32_t>(iter.shape()[d]));
  }
  for (int i = 0; i < 3; ++i) {
    if (i != 0 && iter.is_cpu_scalar(i)) return false;
    for (int d = 0; d < iter.ndim(); ++d) {
      if (iter.strides(i)[d] < 0) return false;
      layout.strides[i][d] = iter.strides(i)[d];
    }
  }
  bool vectorized = iter.numel() > 262144 && iter.shape()[0] % pack == 0;
  const int64_t sizes[3] = {sizeof(res_t), sizeof(storage0_t), sizeof(storage1_t)};
  for (int i = 0; i < 3; ++i) {
    if (i == 0 || layout.strides[i][0] == sizes[i]) {
      vectorized &= is_vector_access_aligned(data[i], sizes[i] * pack,
          {layout.strides[i][1], layout.strides[i][2]});
    }
  }
  const int vt = vectorized ? pack : 1;
  const int64_t blocks = (iter.numel() + 256 * vt - 1) / (256 * vt);
  if (blocks > gpu_max_grid_size(0)) return false;
  const auto stream = at::cuda::getCurrentCUDAStream();
  if (vectorized) {
    elementwise_kernel_binary_packed<pack, storage0_t, storage1_t><<<blocks, 256, 0, stream>>>(
        static_cast<uint32_t>(iter.numel()), data[0], data[1], data[2], layout, f);
  } else {
    elementwise_kernel_binary_packed<1, storage0_t, storage1_t><<<blocks, 256, 0, stream>>>(
        static_cast<uint32_t>(iter.numel()), data[0], data[1], data[2], layout, f);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  log_elementwise_info(iter, "p_e_gpu_binary_packed", f);
  return true;
}

template <int vt, typename func_t>
C10_LAUNCH_BOUNDS_1(256)
__global__ void elementwise_kernel_unary_packed(
    uint32_t N, char* out, const char* in, GPUPackedLayout layout, func_t f) {
  using traits = function_traits<func_t>;
  using res_t = typename traits::result_type;
  using arg_t = typename traits::template arg<0>::type;
  const uint32_t index = (blockIdx.x * blockDim.x + threadIdx.x) * vt;
  if (index >= N) return;
  const auto x = layout.sizes[0].divmod(index);
  const auto y = layout.sizes[1].divmod(x.div);
  const int64_t out_offset = x.mod * layout.strides[0][0] +
      y.mod * layout.strides[0][1] + y.div * layout.strides[0][2];
  const int64_t in_offset = x.mod * layout.strides[1][0] +
      y.mod * layout.strides[1][1] + y.div * layout.strides[1][2];
  const auto input = gpu_packed_load<arg_t, vt>(in, in_offset, layout.strides[1][0]);
  memory::aligned_vector<res_t, vt> result;
#pragma unroll
  for (int i = 0; i < vt; ++i) result.val[i] = f(input.val[i]);
  *reinterpret_cast<memory::aligned_vector<res_t, vt>*>(out + out_offset) = result;
}

template <typename func_t, typename array_t>
static inline bool try_launch_gpu_unary_packed(
    TensorIteratorBase& iter, const func_t& f, array_t data) {
  using traits = function_traits<func_t>;
  using res_t = typename traits::result_type;
  using arg_t = typename traits::template arg<0>::type;
  static_assert(traits::arity == 1);
  if ((iter.ndim() != 2 && iter.ndim() != 3) || iter.ntensors() != 2 ||
      iter.noutputs() != 1 || iter.numel() <= 0 || !iter.can_use_32bit_indexing() ||
      iter.is_cpu_scalar(1) || iter.strides(0)[0] != static_cast<int64_t>(sizeof(res_t))) {
    return false;
  }
  GPUPackedLayout layout{};
  for (int d = 0; d < 2; ++d) {
    layout.sizes[d] = c10::IntDivider<uint32_t>(static_cast<uint32_t>(iter.shape()[d]));
  }
  for (int i = 0; i < 2; ++i) {
    for (int d = 0; d < iter.ndim(); ++d) {
      if (iter.strides(i)[d] < 0) return false;
      layout.strides[i][d] = iter.strides(i)[d];
    }
  }
  bool vectorized = iter.numel() > 262144 && iter.shape()[0] % 4 == 0;
  const int64_t sizes[2] = {sizeof(res_t), sizeof(arg_t)};
  for (int i = 0; i < 2; ++i) {
    if (i == 0 || layout.strides[i][0] == sizes[i]) {
      vectorized &= is_vector_access_aligned(data[i], sizes[i] * 4,
          {layout.strides[i][1], layout.strides[i][2]});
    }
  }
  const int vt = vectorized ? 4 : 1;
  const int64_t blocks = (iter.numel() + 256 * vt - 1) / (256 * vt);
  if (blocks > gpu_max_grid_size(0)) return false;
  const auto stream = at::cuda::getCurrentCUDAStream();
  if (vectorized) {
    elementwise_kernel_unary_packed<4><<<blocks, 256, 0, stream>>>(
        static_cast<uint32_t>(iter.numel()), data[0], data[1], layout, f);
  } else {
    elementwise_kernel_unary_packed<1><<<blocks, 256, 0, stream>>>(
        static_cast<uint32_t>(iter.numel()), data[0], data[1], layout, f);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  log_elementwise_info(iter, "p_e_gpu_unary_packed", f);
  return true;
}

// Shared contract:
//   - vector operands are aligned and contiguous on dim0;
//   - vector accesses do not cross row boundaries;
//   - stride-0 operands use scalar loads;
//   - mismatches fall through to the legacy path.

template <typename func_t, typename array_t>
static inline bool try_launch_gpu_nocast_elementwise_kernel(TensorIteratorBase& iter, const func_t& f, array_t data) {
  if (!elementwise_gpu_enabled()) return false;
  using traits = function_traits<func_t>;
  if constexpr (traits::arity == 2) {
    using res_t = typename traits::result_type;
    using arg0_t = typename traits::template arg<0>::type;
    using arg1_t = typename traits::template arg<1>::type;
    if (iter.ndim() == 3 && sizeof(res_t) >= 8) return false;
    if constexpr (sizeof(res_t) == 8 && sizeof(arg0_t) == 8 && sizeof(arg1_t) == 8) {
      if (iter.ndim() == 2 && iter.numel() <= 1048576 &&
          ((iter.strides(1)[0] == 0 && iter.strides(2)[1] == 0) ||
           (iter.strides(2)[0] == 0 && iter.strides(1)[1] == 0))) {
        return try_launch_gpu_binary_packed<arg0_t, arg1_t>(iter, f, data);
      }
    }
    if constexpr ((sizeof(res_t) == 2 || sizeof(res_t) == 4) &&
                  sizeof(arg0_t) == sizeof(res_t) && sizeof(arg1_t) <= sizeof(res_t)) {
      const int ndim = iter.ndim();
      if (ndim == 2 || ndim == 3) {
        const int64_t width = iter.shape()[0];
        const int64_t s01 = iter.strides(1)[0];
        const int64_t s02 = iter.strides(2)[0];
        const bool narrow_strip = ndim == 2 &&
            ((s01 == 0 && width <= 256) ||
             (width <= 128 && (iter.strides(1)[1] == 0 || iter.strides(2)[1] == 0)));
        // Reordered dim1-contiguous inputs stop amortizing packed divmod and
        // scalar gather work above 1M elements. Keep the proven small/medium
        // domain; larger reordered layouts fall through to legacy.
        const bool reordered = ndim == 2 && iter.numel() <= 1048576 &&
            s02 > static_cast<int64_t>(sizeof(arg1_t));
        const bool large_reordered = ndim == 2 && iter.numel() > 1048576 &&
            s02 > static_cast<int64_t>(sizeof(arg1_t)) && !narrow_strip;
        if (large_reordered) {
          // Do not let a rejected large packed gather be recaptured by the
          // later scalar-gather 1dc path, which is slower for this layout.
          return false;
        }
        const bool narrow_batch = ndim == 3 &&
            (width <= 32 || (s02 == 0 &&
             (iter.numel() <= 1048576 || iter.strides(2)[2] == 0)));
        if (narrow_strip || reordered || narrow_batch) {
          return try_launch_gpu_binary_packed<arg0_t, arg1_t>(iter, f, data);
        }
      }
    }
  }
  if constexpr (traits::arity == 1 && std::is_same_v<typename traits::result_type, float>) {
    const int ndim = iter.ndim();
    const int64_t N = iter.numel();
    if ((ndim == 4 && N <= 65536) || (ndim == 5 && N <= 262144)) return false;
    if (ndim == 3 && iter.strides(1)[0] == sizeof(float) &&
        (N <= 65536 || (iter.shape()[0] <= 32 && iter.strides(1)[1] == 0))) return false;
    if (ndim == 2 && N > 1024 &&
        (N <= 16384 || (N <= 65536 && iter.strides(1)[0] == 0))) return false;
  }
  if constexpr (traits::arity == 1) {
    if constexpr (std::is_same_v<typename traits::result_type, c10::BFloat16> &&
                  std::is_same_v<typename traits::template arg<0>::type, c10::BFloat16>) {
      const int ndim = iter.ndim();
      const int64_t N = iter.numel();
      if ((ndim == 3 && N > 16384 && N <= 1048576 &&
           iter.strides(1)[0] == static_cast<int64_t>(sizeof(c10::BFloat16)) &&
           iter.strides(1)[1] != 0) ||
          (ndim == 2 && N > 1024 && N <= 16384 &&
           (iter.strides(1)[0] == 0 || iter.strides(1)[1] == 0))) {
        return try_launch_gpu_unary_packed(iter, f, data);
      }
    }
  }
  GPU_NOCAST_DISPATCH_TABLE(GPU_DISPATCH_ONE)
  return false;
}

template <typename func_t, typename array_t>
static inline bool try_launch_gpu_cast_elementwise_kernel(TensorIteratorBase& iter, const func_t& f, array_t data) {
  if (!elementwise_gpu_enabled()) return false;
  using traits = function_traits<func_t>;
  using res_t = typename traits::result_type;
  if (iter.ndim() == 3 &&
      (sizeof(res_t) >= 16 || iter.numel() <= 1048576)) return false;
  if constexpr (traits::arity == 2 && std::is_same_v<res_t, float>) {
    if (iter.ndim() == 2 && iter.numel() > 262144 && iter.numel() <= 1048576 &&
        iter.strides(2)[0] == 0 && iter.dtype(1) == ScalarType::BFloat16) return false;
  }
  if constexpr (traits::arity == 2) {
    if constexpr (std::is_same_v<res_t, float> &&
                  std::is_same_v<typename traits::template arg<0>::type, float> &&
                  std::is_same_v<typename traits::template arg<1>::type, float>) {
      if (iter.ndim() == 2 && iter.numel() <= 262144 &&
          iter.dtype(0) == ScalarType::Float && iter.dtype(1) == ScalarType::Float &&
          iter.dtype(2) == ScalarType::BFloat16 &&
          iter.strides(1)[0] == static_cast<int64_t>(sizeof(float)) &&
          iter.strides(2)[0] == static_cast<int64_t>(sizeof(c10::BFloat16)) &&
          iter.strides(2)[1] == 0) {
        return try_launch_gpu_binary_packed<float, c10::BFloat16>(iter, f, data);
      }
    }
  }
  GPU_CAST_DISPATCH_TABLE(GPU_DISPATCH_ONE)
  return false;
}

template <typename func_t, typename array_t>
static inline bool try_launch_gpu_cast_contiguous_unrolled_kernel(TensorIteratorBase& iter, const func_t& f, array_t data) {
  if (!elementwise_gpu_enabled()) return false;
  using traits = function_traits<func_t>;
  GPU_CAST_CONTIG_DISPATCH_TABLE(GPU_DISPATCH_ONE)
  return false;
}
