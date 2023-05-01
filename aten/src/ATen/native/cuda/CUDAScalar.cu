#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/Dispatch.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_local_scalar_dense_native.h>
#endif

#include <ATen/cuda/CUDAContext.h>

namespace at::native {

Scalar _local_scalar_dense_cuda(const Tensor& self) {
  Scalar r;
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND4(
    kComplexHalf, kHalf, kBool, kBFloat16, self.scalar_type(), "_local_scalar_dense_cuda", [&] {
        scalar_t value;
        cudaStream_t stream = at::cuda::getCurrentCUDAStream();
        at::cuda::memcpy_and_sync(&value, self.const_data_ptr<scalar_t>(), sizeof(scalar_t), cudaMemcpyDeviceToHost, stream);
        r = Scalar(value);
      });
  return r;
}

} // at::native
