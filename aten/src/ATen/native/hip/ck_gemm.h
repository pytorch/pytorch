#pragma once

#include <ATen/OpMathType.h>
#include <ATen/hip/HIPBlas.h>
namespace at::native {


template <typename Dtype>
inline void gemm_internal_ck(CUDABLAS_GEMM_ARGTYPES(Dtype)) {
  static_assert(false&&sizeof(Dtype),"at::cuda::blas_gemm_internal_ck: not implemented");
}

template <>
void gemm_internal_ck<double>(CUDABLAS_GEMM_ARGTYPES(double));
template <>
void gemm_internal_ck<float>(CUDABLAS_GEMM_ARGTYPES(float));
template <>
void gemm_internal_ck<at::Half>(CUDABLAS_GEMM_ARGTYPES(at::Half));
template <>
void gemm_internal_ck<at::BFloat16>(CUDABLAS_GEMM_ARGTYPES(at::BFloat16));



} // namespace at::native
