#pragma once

#include <ATen/OpMathType.h>
#include <ATen/hip/HIPBlas.h>

namespace at::native {

template <typename Dtype>
inline void bgemm_internal_ck(CUDABLAS_BGEMM_ARGTYPES(Dtype)) {
  static_assert(false&&sizeof(Dtype),"at::cuda::blas_bgemm_internal_ck: not implemented");
}

template <>
void bgemm_internal_ck<at::BFloat16>(CUDABLAS_BGEMM_ARGTYPES(at::BFloat16));

} // namespace at::native
