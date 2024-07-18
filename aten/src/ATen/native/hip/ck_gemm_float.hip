#undef __HIP_NO_HALF_CONVERSIONS__

#include <ATen/native/hip/ck_gemm.h>
#include <ATen/native/hip/ck_gemm_template.h>

namespace at::native {

void dispatch_float_gemm(CUDABLAS_GEMM_ARGTYPES(float)) {
  // If any of the shapes cant be tiled, we must use padding.
  bool use_padding = ((m % 256 != 0) || (n % 128 != 0) || (k % 64 != 0));
  // Dispatch to best implementation.
  // TODO add more configurations. Optimize.
  if (use_padding) {
    if (m <= 128) {
      gemm_impl<float, 256, 256, 128, 32, 4, 4, 32, 32, 4, 2, 1, true>(CUDABLAS_GEMM_ARGS(float));
    } else {
      gemm_impl<float, 256, 256, 128, 32, 4, 4, 32, 32, 4, 2, 1, true>(CUDABLAS_GEMM_ARGS(float));
    }
  } else {
    {
      gemm_impl<float, 256, 256, 128, 32, 4, 4, 32, 32, 4, 2, 1, false>(CUDABLAS_GEMM_ARGS(float));
    }
  }
}

template <>
void gemm_internal_ck<float>(CUDABLAS_GEMM_ARGTYPES(float)) {
  dispatch_float_gemm(CUDABLAS_GEMM_ARGS(float));
}

// temporarily put this here until we implement double support
template <>
void gemm_internal_ck<double>(CUDABLAS_GEMM_ARGTYPES(double)) {
  return;
}

} // namespace at::native
