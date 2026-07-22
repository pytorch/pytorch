#include <ATen/cuda/CUDAContext.h>
#include <ATen/detail/CUDAHooksInterface.h>
#include <ATen/native/cuda/ScaledBlasDeviceUtils.h>

namespace at::native::scaled {

#ifdef USE_ROCM
bool rocm_scaled_mm_arch_allowed() {
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
}
#else
bool cuda_scaled_mm_arch_allowed(std::initializer_list<CudaScaledMmArch> required_archs) {
  auto dprops = at::cuda::getCurrentDeviceProperties();
  if (required_archs.size() == 0) {
    return dprops->major >= 9 || (dprops->major == 8 && dprops->minor == 9);
  }
  for (auto arch : required_archs) {
    if (dprops->major == static_cast<int64_t>(arch)) {
      return true;
    }
  }
  return false;
}
#endif

}  // namespace at::native::scaled
