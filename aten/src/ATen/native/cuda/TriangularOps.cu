#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/ceil_div.h>
#include <ATen/Context.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/Dispatch.h>
#include <ATen/MemoryOverlap.h>
#include <ATen/native/Resize.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/diag.h>
#include <ATen/ops/diag_native.h>
#include <ATen/ops/trace_native.h>
#include <ATen/ops/tril_native.h>
#include <ATen/ops/triu_native.h>
#endif

#include <ATen/cuda/CUDAApplyUtils.cuh>

#define BOOL_SWITCH(COND, CONST_NAME, ...)      \
  [&] {                                         \
    if (COND) {                                 \
      constexpr static bool CONST_NAME = true;  \
      return __VA_ARGS__();                     \
    } else {                                    \
      constexpr static bool CONST_NAME = false; \
      return __VA_ARGS__();                     \
    }                                           \
  }()

namespace at::native {

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ triu/tril ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

constexpr static int block_size = 128;

template <typename scalar_t, typename IndexType, bool upper, int elements_per_thread, bool inplace>
C10_LAUNCH_BOUNDS_1(block_size)
__global__ void triu_tril_kernel(
    cuda::detail::TensorInfo<scalar_t, IndexType> result_info,
    const cuda::detail::TensorInfo<const scalar_t, IndexType> self_info,
    const int64_t k,
    const int64_t N_padded,
    const IndexType last_dim_padded) {
#if !defined(USE_ROCM)
  int64_t linear_idx = (((int64_t)blockIdx.x) * blockDim.x + threadIdx.x) * elements_per_thread;
  if (linear_idx >= N_padded) {
    return;
  }
#else
  // ROCm limits the total number of threads at 2^32 - use a strided loop to stay within this limit
  for (int64_t linear_idx_retained = (((int64_t) blockIdx.x) * blockDim.x + threadIdx.x) * elements_per_thread;
       linear_idx_retained < N_padded;
       linear_idx_retained += blockDim.x * gridDim.x * elements_per_thread)
  {
    int64_t linear_idx { linear_idx_retained }; // linear_idx_retained persists for the next iteration
#endif // !defined(USE_ROCM)

    auto dims = self_info.dims;

    // Compute column index and row index
    IndexType col = linear_idx % last_dim_padded;
    linear_idx /= last_dim_padded;
    IndexType row = linear_idx % self_info.sizes[dims - 2];

    if constexpr (inplace) {
      bool mask_all_true = upper ? (col - row >= k) : (col + elements_per_thread - row <= k);
      if (mask_all_true)
#if !defined(USE_ROCM)
        return;
#else
        // The strided loop must proceed on ROCm
        continue;
#endif
    }

    // Compute offset
    IndexType self_offset = 0, result_offset = 0;
    self_offset += self_info.strides[dims - 1] * col;
    result_offset += result_info.strides[dims - 1] * col;
    linear_idx /= self_info.sizes[dims - 2];
    self_offset += self_info.strides[dims - 2] * row;
    result_offset += result_info.strides[dims - 2] * row;

    // Compute remaining offsets
    IndexType running_index;
    #pragma unroll
    for (IndexType i = dims - 3; i >= 0; --i) {
      running_index = linear_idx % self_info.sizes[i];
      linear_idx /= self_info.sizes[i];
      self_offset += running_index * self_info.strides[i];
      result_offset += running_index * result_info.strides[i];
    }

    if constexpr (inplace) {
      #pragma unroll
      for (int i = 0; i < elements_per_thread && col + i < self_info.sizes[dims - 1]; i++) {
        bool mask = upper ? (col + i - row >= k) : (col + i - row <= k);
        if (!mask)
          result_info.data[result_offset + i * result_info.strides[dims - 1]] = scalar_t(0);
      }
    } else {
      scalar_t frag[elements_per_thread] = {};
      bool has_mask = (upper && col + elements_per_thread - row >= k) || (!upper && col - row <= k);
      if (has_mask) {
        #pragma unroll
        for (int i = 0; i < elements_per_thread && col + i < self_info.sizes[dims - 1]; i++)
          frag[i] = self_info.data[self_offset + i * self_info.strides[dims - 1]];

        #pragma unroll
        for (int i = 0; i < elements_per_thread; i++) {
          bool mask = upper ? (col + i - row >= k) : (col + i - row <= k);
          frag[i] = mask ? frag[i] : scalar_t(0);
        }
      }

      #pragma unroll
      for (int i = 0; i < elements_per_thread && col + i < self_info.sizes[dims - 1]; i++)
        result_info.data[result_offset + i * result_info.strides[dims - 1]] = frag[i];
    }
#if defined(USE_ROCM)
  } // end of the strided loop on ROCm
#endif // !defined(USE_ROCM)
}

// Helper extracted from the AT_DISPATCH lambda below. The USE_ROCM
// preprocessor branches must live outside the AT_DISPATCH macro expansion;
// MSVC's traditional preprocessor does not accept #if/#else/#endif inside a
// macro argument and the Windows build would otherwise fail.
template <bool upper, typename scalar_t>
void launch_triu_tril_kernel(const Tensor& result, const Tensor& self, int64_t k) {
#if !defined(USE_ROCM)
  constexpr int elements_per_thread = sizeof(scalar_t) < 8 ? 8 / sizeof(scalar_t) : 1;
#else
  // Pair smaller scalars with more elements per thread so the bounded grid
  // (see dim_grid below) still covers the full output via the kernel's
  // strided loop.
  constexpr int elements_per_thread =
    sizeof(scalar_t) <= 2 ? 4 :
    sizeof(scalar_t) <= 8 ? 2 : 1;
#endif
  auto sizes = self.sizes();
  int64_t last_dim_padded = round_up<int64_t>(sizes.back(), elements_per_thread);
  int64_t N_padded = c10::multiply_integers(sizes.begin(), sizes.end() - 1) * last_dim_padded;
  dim3 dim_block = block_size;

#if !defined(USE_ROCM)
  dim3 dim_grid((N_padded / elements_per_thread + dim_block.x - 1) / dim_block.x);
#else
  // ROCm caps total threads (grid * block) at 2^32; bound the grid here and
  // let the kernel walk the remainder via a strided loop so large matrices
  // with 64-bit indexing produce correct results instead of silently
  // truncating.
  const int num_mp = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  constexpr int grid_multiplier = sizeof(scalar_t) <= 2 ? 16 : 32;
  dim3 dim_grid(num_mp * grid_multiplier);
#endif

  if (cuda::detail::canUse32BitIndexMath(result) && cuda::detail::canUse32BitIndexMath(self)) {
    auto result_info = cuda::detail::getTensorInfo<scalar_t, int32_t>(result);
    auto self_info = cuda::detail::getTensorInfo<const scalar_t, int32_t>(self);
    BOOL_SWITCH(self.is_same(result), inplace, [&] {
      triu_tril_kernel<scalar_t, int32_t, upper, elements_per_thread, inplace>
        <<<dim_grid, dim_block, 0, at::cuda::getCurrentCUDAStream()>>>(
          result_info, self_info, k, N_padded, last_dim_padded);
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  } else {
    auto result_info = cuda::detail::getTensorInfo<scalar_t, int64_t>(result);
    auto self_info = cuda::detail::getTensorInfo<const scalar_t, int64_t>(self);
    BOOL_SWITCH(self.is_same(result), inplace, [&] {
      triu_tril_kernel<scalar_t, int64_t, upper, elements_per_thread, inplace>
        <<<dim_grid, dim_block, 0, at::cuda::getCurrentCUDAStream()>>>(
          result_info, self_info, k, N_padded, last_dim_padded);
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
}

template <bool upper>
void triu_tril_cuda_template(const Tensor& result, const Tensor& self, int64_t k, const char* name) {
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND4(
      at::ScalarType::ComplexHalf,
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      at::ScalarType::Bool,
      self.scalar_type(), "triu_tril_cuda_template", [&] {
    launch_triu_tril_kernel<upper, scalar_t>(result, self, k);
  });
}

TORCH_IMPL_FUNC(tril_cuda)(const Tensor& self, int64_t k, const Tensor &result) {
  if (self.numel() != 0) {
    triu_tril_cuda_template<false>(result, self, k, "tril");
  }
}

TORCH_IMPL_FUNC(triu_cuda)(const Tensor& self, int64_t k, const Tensor &result) {
  if (self.numel() != 0) {
    triu_tril_cuda_template<true>(result, self, k, "triu");
  }
}

Tensor trace_cuda(const Tensor& self) {
  TORCH_CHECK(self.dim() == 2, "expected a matrix");
  return self.diagonal().sum();
}

} // namespace at::native
