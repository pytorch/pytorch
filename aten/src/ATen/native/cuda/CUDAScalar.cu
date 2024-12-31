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

#if defined(USE_ROCM)
// TODO(lufang): Tensor.item() on AMD HIP is not synced in the Recsys models.
// This is just a short term workaround. Issue is tracked as FBA-388 on the AMD side.
namespace {
  bool use_sync_mode() {
    static const bool sync_mode = c10::utils::check_env("HIP_DOUBLE_SYNC_ON_LOCAL_SCALE_DENSE") == true;
    return sync_mode;
  }
}
#endif

namespace at::native {

Scalar _local_scalar_dense_cuda(const Tensor& self) {
  Scalar r;
#if defined(USE_ROCM)
  if (!use_sync_mode()){
#endif
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
#if defined(USE_ROCM)
  } else {
    auto cpu_self = self.cpu();
    AT_DISPATCH_V2(
      self.scalar_type(), "_local_scalar_dense_hip", AT_WRAP([&] {
          r = Scalar(*cpu_self.const_data_ptr<scalar_t>());
        }), AT_EXPAND(AT_ALL_TYPES_AND_COMPLEX), kComplexHalf, kHalf, kBool, kBFloat16, AT_EXPAND(AT_BAREBONES_UNSIGNED_TYPES));
  }
#endif
  return r;
}

} // at::native
