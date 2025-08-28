#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/Dispatch_v2.h>
#include <ATen/EmptyTensor.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_local_scalar_dense_native.h>
#endif

#include <ATen/cuda/CUDAContext.h>

namespace at::native {

Scalar _local_scalar_dense_cuda(const Tensor& self) {
  Scalar r;
  TORCH_CHECK(self.numel() > 0, "_local_scalar_dense: Empty tensor not supported");
    AT_DISPATCH_V2(
      self.scalar_type(), "_local_scalar_dense_cuda", AT_WRAP([&] {
          // Create pinned memory for the scalar value to avoid implicit
          // locking/sync in cuda library due to pageable memory
          auto value = at::detail::empty_cpu(
            {1}, /* size */
            c10::CppTypeToScalarType<scalar_t>(), /* dtype */
            std::nullopt, /* layout */
            std::nullopt, /* device */
            true, /* pin_memory */
            std::nullopt /* memory format */
          );
          cudaStream_t stream = at::cuda::getCurrentCUDAStream();
          at::cuda::memcpy_and_sync((void *)value.const_data_ptr<scalar_t>(), self.const_data_ptr<scalar_t>(), sizeof(scalar_t), cudaMemcpyDeviceToHost, stream);
          r = Scalar(*value.const_data_ptr<scalar_t>());
        }), AT_EXPAND(AT_ALL_TYPES_AND_COMPLEX), kComplexHalf, kHalf, kBool, kBFloat16, AT_EXPAND(AT_FLOAT8_TYPES), AT_EXPAND(AT_BAREBONES_UNSIGNED_TYPES));
  return r;
}

} // at::native
