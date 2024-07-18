#undef __HIP_NO_HALF_CONVERSIONS__

#include <ATen/native/hip/ck_gemm.h>
#include <ATen/native/hip/ck_gemm_template.h>

namespace at::native {

void dispatch_bfloat16_gemm(CUDABLAS_GEMM_ARGTYPES(at::BFloat16)) {
  // If any of the shapes cant be tiled, we must use padding.
  bool use_padding = ((m % 256 != 0) || (n % 128 != 0) || (k % 64 != 0));
  // Dispatch to best implementation.
  // TODO add more configurations. Optimize.
  if (use_padding) {
    if (m <= 128) {
      gemm_impl<at::BFloat16, 256, 256, 128, 32, 4, 4, 32, 32, 4, 2, 1, true>(CUDABLAS_GEMM_ARGS(at::BFloat16));
    } else {
      gemm_impl<at::BFloat16, 256, 256, 128, 32, 4, 4, 32, 32, 4, 2, 1, true>(CUDABLAS_GEMM_ARGS(at::BFloat16));
    }
  } else {
    {
      gemm_impl<at::BFloat16, 256, 256, 128, 32, 4, 4, 32, 32, 4, 2, 1, false>(CUDABLAS_GEMM_ARGS(at::BFloat16));
    }
  }
}

template <>
void gemm_internal_ck<at::BFloat16>(CUDABLAS_GEMM_ARGTYPES(at::BFloat16)) {
  dispatch_bfloat16_gemm(CUDABLAS_GEMM_ARGS(at::BFloat16));
}

} // namespace at::native
