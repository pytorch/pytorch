#include <cuda_runtime.h>

#include <torch/csrc/distributed/c10d/cuda/utils.hpp>

#if !defined(USE_ROCM) && defined(PYTORCH_C10_DRIVER_API_SUPPORTED)
#include <c10/cuda/driver_api.h>
#endif

#if defined(CUDART_VERSION) && CUDART_VERSION >= 12030
#define CUDART_SUPPORTS_MULTICAST
#endif

namespace c10d::cuda {

bool deviceSupportsMulticast(int device_idx) {
#if defined(CUDART_SUPPORTS_MULTICAST)
  // Multicast support requirements:
  // - CUDA Runtime version >= 12030: Checked at compile time using
  // CUDART_VERSION.
  // - Driver version >= 535: Checked at runtime by verifying the existence of
  // cuMulticastCreate_.
  // - Device support: Determined by querying
  // CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED at runtime.
  auto driver_api = c10::cuda::DriverAPI::get();
  int multicast_supported = 0;
  C10_CUDA_DRIVER_CHECK(driver_api->cuDeviceGetAttribute_(
      &multicast_supported,
      CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED,
      device_idx));
  return driver_api->cuMulticastCreate_ != nullptr && multicast_supported;
#else
  return false;
#endif
}

} // namespace c10d::cuda
