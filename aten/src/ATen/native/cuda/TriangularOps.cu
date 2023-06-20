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

namespace at::native {

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ triu/tril ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

template <typename scalar_t, typename IndexType, bool upper>
C10_LAUNCH_BOUNDS_1(cuda::getApplyBlockSize())
__global__ void triu_tril_kernel(
    cuda::detail::TensorInfo<scalar_t, IndexType> result_info,
    const cuda::detail::TensorInfo<scalar_t, IndexType> self_info,
    const int64_t k,
    const int64_t N) {
  int64_t linear_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (linear_idx >= N) {
    return;
  }

  auto dims = self_info.dims;

  IndexType self_offset = 0, result_offset = 0;
  // Compute column index and corresponding offset
  IndexType col = linear_idx % self_info.sizes[dims - 1];
  linear_idx /= self_info.sizes[dims - 1];
  self_offset += self_info.strides[dims - 1] * col;
  result_offset += result_info.strides[dims - 1] * col;

  // Compute row index and corresponding offset
  IndexType row = linear_idx % self_info.sizes[dims - 2];
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

  bool mask = upper ? (col - row >= k) : (col - row <= k);
  result_info.data[result_offset] = mask ? self_info.data[self_offset] : scalar_t(0);
}

template <bool upper>
void triu_tril_cuda_template(const Tensor& result, const Tensor& self, int64_t k, const char* name) {
  int64_t N = self.numel();
  dim3 dim_block = cuda::getApplyBlock();
  dim3 dim_grid((N + dim_block.x - 1) / dim_block.x);
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND4(
      at::ScalarType::ComplexHalf,
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      at::ScalarType::Bool,
      self.scalar_type(), "triu_tril_cuda_template", [&] {
    if (cuda::detail::canUse32BitIndexMath(result) && cuda::detail::canUse32BitIndexMath(self)) {
      auto result_info = cuda::detail::getTensorInfo<scalar_t, int32_t>(result);
      auto self_info = cuda::detail::getTensorInfo<scalar_t, int32_t>(self);
      triu_tril_kernel<scalar_t, int32_t, upper>
        <<<dim_grid, dim_block, 0, at::cuda::getCurrentCUDAStream()>>>(
          result_info, self_info, k, N);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    } else {
      auto result_info = cuda::detail::getTensorInfo<scalar_t, int64_t>(result);
      auto self_info = cuda::detail::getTensorInfo<scalar_t, int64_t>(self);
      triu_tril_kernel<scalar_t, int64_t, upper>
        <<<dim_grid, dim_block, 0, at::cuda::getCurrentCUDAStream()>>>(
          result_info, self_info, k, N);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
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
