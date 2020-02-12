#include <ATen/Context.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/Dispatch.h>
#include <ATen/NativeFunctions.h>

#include <ATen/cuda/CUDAApplyUtils.cuh>
#include <ATen/cuda/detail/IndexUtils.cuh>

namespace at {
namespace native {

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ triu/tril ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

template <typename scalar_t, typename IndexType, bool upper>
#ifdef __HIP_PLATFORM_HCC__
C10_LAUNCH_BOUNDS_1(512)
#endif
__global__
void triu_tril_kernel(
    cuda::detail::TensorInfo<scalar_t, IndexType> result_info,
    const cuda::detail::TensorInfo<scalar_t, IndexType> self_info,
    const int64_t k, const int64_t N) {
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
Tensor& triu_tril_cuda_template(Tensor& result, const Tensor& self, int64_t k, const char* name) {
  int64_t N = self.numel();
  dim3 dim_block = cuda::getApplyBlock();
  dim3 dim_grid((N + dim_block.x - 1) / dim_block.x);
  AT_DISPATCH_ALL_TYPES_AND2(at::ScalarType::Half, at::ScalarType::Bool, self.scalar_type(), name, [&]{
    if (cuda::detail::canUse32BitIndexMath(result) && cuda::detail::canUse32BitIndexMath(self)) {
      auto result_info = cuda::detail::getTensorInfo<scalar_t, int32_t>(result);
      auto self_info = cuda::detail::getTensorInfo<scalar_t, int32_t>(self);
      triu_tril_kernel<scalar_t, int32_t, upper>
        <<<dim_grid, dim_block, 0, at::cuda::getCurrentCUDAStream()>>>(
          result_info, self_info, k, N);
    } else {
      auto result_info = cuda::detail::getTensorInfo<scalar_t, int64_t>(result);
      auto self_info = cuda::detail::getTensorInfo<scalar_t, int64_t>(self);
      triu_tril_kernel<scalar_t, int64_t, upper>
        <<<dim_grid, dim_block, 0, at::cuda::getCurrentCUDAStream()>>>(
          result_info, self_info, k, N);
    }
  });
  AT_CUDA_CHECK(cudaGetLastError());
  return result;
}

Tensor& tril_cuda_(Tensor &self, int64_t k) {
  return tril_cuda_out(self, self, k);
}

Tensor& tril_cuda_out(Tensor &result, const Tensor& self, int64_t k) {
  if (result.sizes() != self.sizes()) {
    result.resize_as_(self);
  }
  if (self.numel() == 0) {
    return result;
  }
  return triu_tril_cuda_template<false>(result, self, k, "tril");
}

Tensor& triu_cuda_(Tensor &self, int64_t k) {
  return triu_cuda_out(self, self, k);
}

Tensor& triu_cuda_out(Tensor &result, const Tensor& self, int64_t k) {
  if (result.sizes() != self.sizes()) {
    result.resize_as_(self);
  }
  if (self.numel() == 0) {
    return result;
  }
  return triu_tril_cuda_template<true>(result, self, k, "triu");
}

}  // namespace native
}  // namespace at
