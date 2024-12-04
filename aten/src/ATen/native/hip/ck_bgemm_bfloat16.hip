#undef __HIP_NO_HALF_CONVERSIONS__

#include <iostream>

#include <ATen/native/hip/ck_bgemm.h>
#include <ATen/native/hip/bgemm_kernels/bgemm_kernel_collection.h>

namespace at::native {

using BGEMMKernel_BFloat16 = std::function<void(CUDABLAS_BGEMM_ARGTYPES(at::BFloat16))>;

// This is the heursitic to choose a kernel based on inputs
BGEMMKernel_BFloat16 dispatch_bfloat16_bgemm(CUDABLAS_BGEMM_ARGTYPES(at::BFloat16)) {
  // just have one instance for now
  return bgemm_kernel_256_256x224x64_16x16_8x7_8x32x1_8x32x1_1x32x1x8_4_intrawave_v3;
}

template <>
void bgemm_internal_ck<at::BFloat16>(CUDABLAS_BGEMM_ARGTYPES(at::BFloat16)) {
  /*
  std::cerr << "bgemm_internal_ck: " << std::endl
   << "\t" << num_batches
   << "\t" << m
   << "\t" << n
   << "\t" << k << std::endl
   << "\t" << stridea
   << "\t" << strideb
   << "\t" << stridec
   << "\t" << lda
   << "\t" << ldb
   << "\t" << ldc << std::endl
   << "\t" << transa
   << "\t" << transb << std::endl
   << "\t" << alpha
   << "\t" << beta << std::endl;
   */

  BGEMMKernel_BFloat16 bgemm_impl = dispatch_bfloat16_bgemm(CUDABLAS_BGEMM_ARGS(at::BFloat16));
  bgemm_impl(CUDABLAS_BGEMM_ARGS(at::BFloat16));
}

} // namespace at::native
