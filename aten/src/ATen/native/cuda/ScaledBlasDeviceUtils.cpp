#include <ATen/cuda/CUDAContext.h>
#include <ATen/detail/CUDAHooksInterface.h>
#include <ATen/native/cuda/ScaledBlasDeviceUtils.h>

#include <string>
#include <vector>

namespace at::native::scaled {

#ifdef USE_ROCM
// On ROCm, sm90_only/sm100_only are ignored.
bool scaled_mm_arch_allowed(
    bool /*sm90_only*/,
    bool /*sm100_only*/,
    c10::DeviceIndex device_index) {
  static const std::vector<std::string> archs = {
      "gfx942",
#if ROCM_VERSION >= 60500
      // gfx950 and gfx120x only support OCP fp8, which requires ROCm 6.5.
      "gfx950", "gfx1200", "gfx1201",
#endif
#if ROCM_VERSION >= 71400
      "gfx1250",
#endif
  };
  return at::detail::getCUDAHooks().isGPUArch(archs, device_index);
}
#else
bool scaled_mm_arch_allowed(
    bool sm90_only,
    bool sm100_only,
    c10::DeviceIndex device_index) {
  auto dprops = at::cuda::getDeviceProperties(device_index);
  if (sm90_only || sm100_only) {
    return (sm90_only && dprops->major == 9) || (sm100_only && dprops->major == 10);
  }
  return dprops->major >= 9 || (dprops->major == 8 && dprops->minor == 9);
}
#endif

} // namespace at::native::scaled
