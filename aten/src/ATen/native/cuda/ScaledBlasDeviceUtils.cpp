#include <ATen/cuda/CUDAContext.h>
#include <ATen/detail/CUDAHooksInterface.h>
#include <ATen/native/cuda/ScaledBlasDeviceUtils.h>

#include <string>
#include <vector>

namespace at::native::scaled {

#ifdef USE_ROCM
// On ROCm, sm90_only/sm100_only are ignored.
bool scaled_mm_arch_allowed(bool /*sm90_only*/, bool /*sm100_only*/) {
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
#if ROCM_VERSION >= 100300
      "gfx1170", "gfx1171",
#endif
  };
  return at::detail::getCUDAHooks().isGPUArch(archs);
}
#else
bool scaled_mm_arch_allowed(bool sm90_only, bool sm100_only) {
  auto dprops = at::cuda::getCurrentDeviceProperties();
  if (sm90_only || sm100_only) {
    return (sm90_only && dprops->major == 9) || (sm100_only && dprops->major == 10);
  }
  return dprops->major >= 9 || (dprops->major == 8 && dprops->minor == 9);
}
#endif

void check_mnk4_support() {
#ifdef USE_ROCM
  TORCH_CHECK_NOT_IMPLEMENTED(false, "packed MNxK4 scaling is not supported on ROCm");
#else
  const auto major = at::cuda::getCurrentDeviceProperties()->major;
  TORCH_CHECK_NOT_IMPLEMENTED(
      major == 10 || major == 11,
      "packed MNxK4 scaling is only supported on SM10.x and SM11.x");
  TORCH_CHECK_NOT_IMPLEMENTED(
      CUBLAS_VERSION >= 130801,
      "packed MNxK4 scaling requires a build with cuBLASLt >= 13.8.1 headers; built with ", CUBLAS_VERSION);
  const auto runtime_version = cublasLtGetVersion();
  TORCH_CHECK_NOT_IMPLEMENTED(
      runtime_version >= 130801,
      "packed MNxK4 scaling requires cuBLASLt runtime >= 13.8.1; got ", runtime_version);
#endif
}

} // namespace at::native::scaled
