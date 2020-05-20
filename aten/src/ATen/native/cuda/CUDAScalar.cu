#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>

#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>

#ifdef __HIP_PLATFORM_HCC__
#include <hip/hip_version.h>
#endif

namespace at {
namespace native {

Scalar _local_scalar_dense_cuda(const Tensor& self) {
  Scalar r;
#if HIP_VERSION >= 301
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
    at::ScalarType::Half, at::ScalarType::Bool, at::ScalarType::BFloat16, self.scalar_type(), "_local_scalar_dense_cuda", [&] {
        scalar_t value;
        cudaStream_t stream = at::cuda::getCurrentCUDAStream();
        AT_CUDA_CHECK(hipMemcpyWithStream(&value, self.data_ptr<scalar_t>(), sizeof(scalar_t), cudaMemcpyDeviceToHost, stream));
        r = Scalar(value);
      });
#else
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
    at::ScalarType::Half, at::ScalarType::Bool, at::ScalarType::BFloat16, self.scalar_type(), "_local_scalar_dense_cuda", [&] {
        scalar_t value;
        cudaStream_t stream = at::cuda::getCurrentCUDAStream();
        AT_CUDA_CHECK(cudaMemcpyAsync(&value, self.data_ptr<scalar_t>(), sizeof(scalar_t), cudaMemcpyDeviceToHost, stream));
        AT_CUDA_CHECK(cudaStreamSynchronize(stream));
        r = Scalar(value);
      });
#endif
  return r;
}

}} // at::native
