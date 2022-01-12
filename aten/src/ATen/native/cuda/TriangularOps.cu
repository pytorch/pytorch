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
#include <ATen/ops/trace_native.h>
#include <ATen/ops/diag_native.h>
#include <ATen/ops/triu_native.h>
#include <ATen/ops/tril_native.h>
#endif

#include <ATen/cuda/CUDAApplyUtils.cuh>

namespace at {
namespace native {

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
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(at::ScalarType::Half, at::ScalarType::Bool, self.scalar_type(), "triu_tril_cuda_template", [&]{
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

// Copy the kth diagonal of a matrix B to a vector A.
template <typename scalar_t>
C10_LAUNCH_BOUNDS_1(1024)
__global__ void copy_from_diagonal_kernel(
    scalar_t* a,
    scalar_t* b,
    std::ptrdiff_t start,
    std::ptrdiff_t size,
    std::ptrdiff_t strideSum,
    std::ptrdiff_t strideA) {
  for (std::ptrdiff_t linearIndex = blockIdx.x * blockDim.x + threadIdx.x;
       linearIndex < size;
       linearIndex += gridDim.x * blockDim.x) {
    const std::ptrdiff_t bOffset = start + strideSum * linearIndex;
    a[strideA * linearIndex] = b[bOffset];
  }
}

// Copy vector B to the kth diagonal of a matrix A
template <typename scalar_t>
C10_LAUNCH_BOUNDS_1(1024)
__global__ void copy_to_diagonal_kernel(
    scalar_t* a,
    scalar_t* b,
    std::ptrdiff_t start,
    std::ptrdiff_t size,
    std::ptrdiff_t strideSum,
    std::ptrdiff_t strideB) {
  for (std::ptrdiff_t linearIndex = blockIdx.x * blockDim.x + threadIdx.x;
       linearIndex < size;
       linearIndex += gridDim.x * blockDim.x) {
    const std::ptrdiff_t aOffset = start + strideSum * linearIndex;
    a[aOffset] = b[strideB * linearIndex];
  }
}

template <typename scalar_t>
Tensor& apply_diag(Tensor& result, const Tensor& self, int64_t dimension) {
  TORCH_CHECK(
      self.dim() == 1 || self.dim() == 2, "matrix or a vector expected");

  TensorArg result_arg{result, "result", 1};
  TensorArg self_arg{self, "self", 2};
  checkAllSameGPU(__func__, {result_arg, self_arg});
  checkSameType(__func__, result_arg, self_arg);

  int nDimension = self.dim();
  if (nDimension == 2) {
    auto self_stride_0 = self.stride(0);
    auto self_stride_1 = self.stride(1);

    int sz;
    if (dimension > 0) {
      sz = std::min(self.size(0), self.size(1) - dimension);
    } else {
      sz = std::min(self.size(0) + dimension, self.size(1));
    }

    at::native::resize_output(result, {sz});
    if (sz > 0) {
      at::assert_no_internal_overlap(result);
      auto result_stride = result.stride(0);
      const dim3 threads(std::min(
          int(sz),
          int(at::cuda::getCurrentDeviceProperties()->maxThreadsPerBlock)));
      const dim3 grid(
          std::min(int(1024), ceil_div(int(sz), int(threads.x))));
      auto start =
          (dimension >= 0 ? dimension * self_stride_1
                          : -dimension * self_stride_0);

      // Kernel Launch
      copy_from_diagonal_kernel<scalar_t>
          <<<grid, threads, 0, c10::cuda::getCurrentCUDAStream()>>>(
              result.data_ptr<scalar_t>(),
              self.data_ptr<scalar_t>(),
              start,
              sz,
              self_stride_0 + self_stride_1,
              result_stride);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
  } else {
    auto n_elems = self.numel();
    auto sz = (dimension > 0) ? n_elems + dimension : n_elems - dimension;
    auto self_stride = self.stride(0);
    at::native::resize_output(result, {sz, sz});
    result.zero_();
    if (sz > 0) {
      at::assert_no_internal_overlap(result);
      auto result_stride_0 = result.stride(0);
      auto result_stride_1 = result.stride(1);
      const dim3 threads(std::min(
          int(sz), at::cuda::getCurrentDeviceProperties()->maxThreadsPerBlock));
      const dim3 grid(
          std::min(int(1024), ceil_div(int(sz), int(threads.x))));
      auto start =
          (dimension >= 0 ? dimension * result_stride_1
                          : -dimension * result_stride_0);

      // Kernel Launch
      copy_to_diagonal_kernel<scalar_t>
          <<<grid, threads, 0, c10::cuda::getCurrentCUDAStream()>>>(
              result.data_ptr<scalar_t>(),
              self.data_ptr<scalar_t>(),
              start,
              n_elems,
              result_stride_0 + result_stride_1,
              self_stride);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
  }

  return result;
}

Tensor& diag_cuda_out(const Tensor& self, int64_t dimension, Tensor& result) {
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
      ScalarType::Half, ScalarType::BFloat16, ScalarType::Bool,
      self.scalar_type(), "diag_cuda",
      [&] {
        apply_diag<scalar_t>(result, self, dimension);
      });
  return result;
}

Tensor trace_cuda(const Tensor& self) {
  TORCH_CHECK(self.dim() == 2, "expected a matrix");
  int dimension = 0;
  auto result = at::diag(self, dimension);
  return result.sum();
}

} // namespace native
} // namespace at
