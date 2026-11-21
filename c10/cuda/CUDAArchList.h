#pragma once

#include <c10/macros/Macros.h>

namespace c10::cuda {

// nvcc defines __CUDA_ARCH_LIST__ in the host pass too. It is per translation
// unit, not per build, and holds virtual architectures, so a +PTX build is not
// credited with its JIT range.
constexpr C10_HOST_DEVICE bool targets_any_arch_in(
    [[maybe_unused]] int sm_lo,
    [[maybe_unused]] int sm_hi) {
#ifdef __CUDA_ARCH_LIST__
  constexpr int archs[] = {__CUDA_ARCH_LIST__};
  for (int arch : archs) {
    if (arch / 10 >= sm_lo && arch / 10 <= sm_hi) {
      return true;
    }
  }
  return false;
#else
  return true;
#endif
}

} // namespace c10::cuda
