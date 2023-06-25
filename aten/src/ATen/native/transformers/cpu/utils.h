#include <ATen/ATen.h>
#include <ATen/cpu/vec/functional.h>
#include <ATen/cpu/vec/vec.h>

#if AT_MKL_ENABLED()
#include <mkl.h>

inline void _mkl_gemm(
    const CBLAS_LAYOUT layout,
    const CBLAS_TRANSPOSE transa,
    const CBLAS_TRANSPOSE transb,
    const int& m,
    const int& n,
    const int& k,
    const float& alpha,
    float* a,
    const int& lda,
    float* b,
    const int& ldb,
    const float& beta,
    float* c,
    const int& ldc) {
  cblas_sgemm(
      layout,
      transa,
      transb,
      m,
      n,
      k,
      alpha,
      a,
      lda,
      b,
      ldb,
      beta,
      c,
      ldc);
}

inline void _mkl_gemm(
    const CBLAS_LAYOUT layout,
    const CBLAS_TRANSPOSE transa,
    const CBLAS_TRANSPOSE transb,
    const int& m,
    const int& n,
    const int& k,
    const float& alpha,
    at::BFloat16* a,
    const int& lda,
    at::BFloat16* b,
    const int& ldb,
    const float& beta,
    float* c,
    const int& ldc) {
  cblas_gemm_bf16bf16f32(
      layout,
      transa,
      transb,
      m,
      n,
      k,
      alpha,
      (MKL_BF16*)(a),
      lda,
      (MKL_BF16*)(b),
      ldb,
      beta,
      c,
      ldc);
}

#endif // AT_MKL_ENABLED

inline void _store(
    float* dst,
    at::vec::Vectorized<float> src,
    int64_t l=at::vec::Vectorized<float>::size()) {
  src.store(dst, l);
}

inline void _store(
    at::BFloat16* dst,
    at::vec::Vectorized<float> src,
    int64_t l=at::vec::Vectorized<float>::size()) {
  auto res = at::vec::convert_float_bfloat16(src, src);
  res.store(dst, l);
}
