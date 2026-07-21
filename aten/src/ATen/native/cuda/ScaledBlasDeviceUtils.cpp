#include <ATen/cuda/CUDAContext.h>
#include <ATen/detail/CUDAHooksInterface.h>
#include <ATen/native/ScaledBlasUtils.h>

namespace at::native::scaled {

bool scaled_mm_allowed_device(bool sm90_only, bool sm100_only) {
#ifdef USE_ROCM
  static const std::vector<std::string> archs = {
      "gfx942",
#if ROCM_VERSION >= 60300
      "gfx1200", "gfx1201",
#endif
#if ROCM_VERSION >= 60500
      "gfx950",
#endif
#if ROCM_VERSION >= 71400
      "gfx1250",
#endif
  };
  return at::detail::getCUDAHooks().isGPUArch(archs);
#else
  auto dprops = at::cuda::getCurrentDeviceProperties();

  if (sm90_only || sm100_only) {
    return (sm90_only && dprops->major == 9) || (sm100_only && dprops->major == 10);
  } else {
    return dprops->major >= 9 || (dprops->major == 8 && dprops->minor == 9);
  }
#endif
}

}  // namespace at::native::scaled
