#pragma once

#include <c10/util/Exception.h>
#include <cutlass/cutlass.h>

#include <initializer_list>

namespace at::cuda::detail {

// Whether nvcc generates device code for any arch in [lo, hi). Also usable in
// the host pass, to skip instantiating kernels no targeted arch gives a body.
// static: the list differs between TUs (see _BUILD_FOR_ADDITIONAL_ARCHS).
static constexpr bool cuda_arch_list_has(int lo, int hi) {
#ifdef __CUDA_ARCH_LIST__
  for (int arch : {__CUDA_ARCH_LIST__}) {
    if (arch >= lo && arch < hi) {
      return true;
    }
  }
  return false;
#else
  return true;
#endif
}

static constexpr bool cuda_arch_list_has_major(int major) {
  return cuda_arch_list_has(major * 100, (major + 1) * 100);
}

[[noreturn]] inline void cutlass_arch_not_built(const char* op) {
  TORCH_CHECK(
      false,
      op,
      " was not built for this device's compute capability; add it to "
      "TORCH_CUDA_ARCH_LIST");
}

template <typename Kernel>
struct enable_2x_kernel_for_sm89 : Kernel {
  template <typename... Args>
  CUTLASS_DEVICE static void invoke(Args&&... args) {
#if defined __CUDA_ARCH__ && __CUDA_ARCH__ == 890
    Kernel::invoke(std::forward<Args>(args)...);
#endif
  }
};

template <typename Kernel>
struct enable_3x_kernel_for_sm9x : Kernel {
  template <typename... Args>
  CUTLASS_DEVICE void operator()(Args&&... args) {
#if defined __CUDA_ARCH__ && __CUDA_ARCH__ >= 900 && __CUDA_ARCH__ < 1000
    Kernel::operator()(std::forward<Args>(args)...);
#endif
  }
};

template <typename Kernel>
struct enable_3x_kernel_for_sm10 : Kernel {
  template <typename... Args>
  CUTLASS_DEVICE void operator()(Args&&... args) {
#if defined __CUDA_ARCH__ && __CUDA_ARCH__ >= 1000 && __CUDA_ARCH__ < 1200
    Kernel::operator()(std::forward<Args>(args)...);
#endif
  }
};

template <typename Kernel>
struct enable_3x_kernel_for_sm10_or_later : Kernel {
  template <typename... Args>
  CUTLASS_DEVICE void operator()(Args&&... args) {
#if defined __CUDA_ARCH__ && __CUDA_ARCH__ >= 1000
    Kernel::operator()(std::forward<Args>(args)...);
#endif
  }
};

}  // namespace at::cuda::detail
