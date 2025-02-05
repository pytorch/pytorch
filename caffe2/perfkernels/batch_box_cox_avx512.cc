#ifdef CAFFE2_PERF_USE_MKL
#include <immintrin.h>

// Enable compiler vectorized version only if numerical consistency is not
// required between dev and opt versions - disabled for now
#ifndef FAST_VECTORIZED_KERNEL
#define CPU_CAPABILITY_AVX512
#include <ATen/cpu/vec/vec.h>

namespace at::vec {
namespace {
// Implements the vectorized version of std::max() operation,
// which DOESNOT propagates NaN for second argument
template <typename scalar_t>
Vectorized<scalar_t> max(const Vectorized<scalar_t>& a, const Vectorized<scalar_t>& b);

template <>
Vectorized<double> max(const Vectorized<double>& a, const Vectorized<double>& b) {
  // std::max(NaN, nonNan) -> NaN
  return _mm512_max_pd(b, a);
}

template <>
Vectorized<float> max(const Vectorized<float>& a, const Vectorized<float>& b) {
  // std::max(NaN, nonNan) -> NaN
  return _mm512_max_ps(b, a);
}

// Implements recieprocal method based on newton-rapson method
// 1. user RCP approximiation
// 2. update with RCP = RCP * (2 - X * RCP)
template <typename scalar_t>
Vectorized<scalar_t> fast_recieprocal(const Vectorized<scalar_t>& b);
template <typename scalar_t>
scalar_t fast_recieprocal(scalar_t b);

template<>
Vectorized<float> fast_recieprocal(const Vectorized<float>& b) {
  auto minus2 = _mm512_set1_ps(-2.f);
  auto rcp = _mm512_rcp14_ps(b);
  rcp = _mm512_mul_ps(rcp,  _mm512_fnmsub_ps(rcp, b, minus2));
  rcp = _mm512_mul_ps(rcp,  _mm512_fnmsub_ps(rcp, b, minus2));
  return rcp;
}

template <>
float fast_recieprocal(float b) {
  auto minus2 = _mm_set_ss(-2.f);
  auto b_reg = _mm_set_ss(b);
  auto rcp = _mm_rcp_ss(b_reg);
  rcp = _mm_mul_ss(rcp,  _mm_fnmsub_ss(rcp, b_reg, minus2));
  rcp = _mm_mul_ss(rcp,  _mm_fnmsub_ss(rcp, b_reg, minus2));
  return _mm_cvtss_f32(rcp);
}

template<>
Vectorized<double> fast_recieprocal(const Vectorized<double>& b) {
  auto minus2 = _mm512_set1_pd(-2.);
  auto rcp = _mm512_rcp14_pd(b);
  rcp = _mm512_mul_pd(rcp,  _mm512_fnmsub_pd(rcp, b, minus2));
  rcp = _mm512_mul_pd(rcp,  _mm512_fnmsub_pd(rcp, b, minus2));
  return rcp;
}

template <>
double fast_recieprocal(double b) {
  return 1./b;
}
} // namespace
} // namespace at::vec
#endif

#include "caffe2/perfkernels/batch_box_cox_vec.h"

namespace caffe2::details {

template <typename T>
void compute_batch_box_cox__avx512(
    std::size_t N,
    std::size_t D,
    std::size_t block_size,
    const T* self_data,
    const T* __restrict lambda1_data,
    const T* __restrict lambda2_data,
    T* output_data) {
      compute_batch_box_cox_vec_fma<T>(
          N,
          D,
          block_size,
          self_data,
          lambda1_data,
          lambda2_data,
          output_data);
    }

// Vectorized version specializations for float and double
template
void compute_batch_box_cox__avx512<float>(
  std::size_t N,
  std::size_t D,
  std::size_t block_size,
  const float* self_data,
  const float* __restrict lambda1_data,
  const float* __restrict lambda2_data,
  float* output_data);

template
void compute_batch_box_cox__avx512<double>(
  std::size_t N,
  std::size_t D,
  std::size_t block_size,
  const double* self_data,
  const double* __restrict lambda1_data,
  const double* __restrict lambda2_data,
  double* output_data);

} // namespace caffe2::detail
#endif // CAFFE2_PERF_USE_MKL
