#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGraphsC10Utils.h>

#include <cuda_runtime.h>

namespace at::native {

bool supports_large_kernel_arg() {
#if !defined(USE_ROCM) && !defined(_WIN32) && defined(CUDART_VERSION) && CUDART_VERSION >= 12010
  static std::optional<bool> supports_large_kernel_arg_ = std::nullopt;
  if (!supports_large_kernel_arg_.has_value()) {
    int driver_ver = 0;
    AT_CUDA_CHECK(cudaDriverGetVersion(&driver_ver));
    cudaDeviceProp* prop = at::cuda::getCurrentDeviceProperties();
    supports_large_kernel_arg_ = (driver_ver >= 12010) && prop->major >= 7;
  }
  const bool is_capturing = at::cuda::currentStreamCaptureStatusMayInitCtx() !=
      at::cuda::CaptureStatus::None;
  return !is_capturing && *supports_large_kernel_arg_;
#else
  return false;
#endif
}

} // namespace at::native
