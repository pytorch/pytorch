#pragma once

#include <c10/util/Exception.h>
#include <cutlass/cutlass.h>

namespace at::cuda::detail {

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
struct enable_3x_kernel_for_sm10_or_later : Kernel {
  template <typename... Args>
  CUTLASS_DEVICE void operator()(Args&&... args) {
#if defined __CUDA_ARCH__ && __CUDA_ARCH__ >= 1000
    Kernel::operator()(std::forward<Args>(args)...);
#endif
  }
};

}  // namespace at::cuda::detail
