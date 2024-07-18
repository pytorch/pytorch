#undef __HIP_NO_HALF_CONVERSIONS__

#include <ATen/native/hip/ck_gemm.h>
#include <ATen/native/hip/ck_gemm_template.h>

namespace at::native {

void dispatch_half_gemm(CUDABLAS_GEMM_ARGTYPES(at::Half)) {
  // If any of the shapes cant be tiled, we must use padding.
  bool use_padding = ((m % 256 != 0) || (n % 128 != 0) || (k % 64 != 0));
  // Dispatch to best implementation.
  // TODO add more configurations. Optimize.
  if (use_padding) {
    if (m <= 128) {
      gemm_impl<at::Half, 256, 256, 128, 32, 4, 4, 32, 32, 4, 2, 1, true>(CUDABLAS_GEMM_ARGS(at::Half));
    } else {
      gemm_impl<at::Half, 256, 256, 128, 32, 4, 4, 32, 32, 4, 2, 1, true>(CUDABLAS_GEMM_ARGS(at::Half));
    }
  } else {
    {
      gemm_impl<at::Half, 256, 256, 128, 32, 4, 4, 32, 32, 4, 2, 1, false>(CUDABLAS_GEMM_ARGS(at::Half));
    }
  }
}

template <>
void gemm_internal_ck<at::Half>(CUDABLAS_GEMM_ARGTYPES(at::Half)) {
  dispatch_half_gemm(CUDABLAS_GEMM_ARGS(at::Half));
}

} // namespace at::native
