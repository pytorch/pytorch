#include "caffe2/perfkernels/common.h"

#include <algorithm>
#include <cstdint>
#include <cmath>

namespace caffe2 {

namespace {
template <typename T>
void BoxCoxNaive(
    std::size_t N,
    std::size_t D,
    const T* data_ptr,
    const T* __restrict lambda1_ptr,
    const T* __restrict lambda2_ptr,
    T* output_ptr) {
  constexpr T k_eps = static_cast<T>(1e-6);

  for (std::size_t i = 0; i < N; i++) {
    for (std::size_t j = 0; j < D; j++, data_ptr++, output_ptr++) {
      T lambda1_v = lambda1_ptr[j];
      T lambda2_v = lambda2_ptr[j];
      T tmp = std::max(*data_ptr + lambda2_v, k_eps);
      if (lambda1_v == 0) {
        *output_ptr = std::log(tmp);
      } else {
        T lambda_1 = 1 / lambda1_v;
        T pow = std::pow(tmp, lambda1_v);
        *output_ptr = lambda_1 * pow - lambda_1;
      }
    }
  }

}
}

#if defined(CAFFE2_PERF_WITH_AVX2) && defined(CAFFE2_PERF_USE_MKL)
namespace details {
template <typename T>
void compute_batch_box_cox__avx2_fma(
  std::size_t N,
  std::size_t D,
  std::size_t block_size,
  const T* data_ptr,
  const T* __restrict lambda1_ptr,
  const T* __restrict lambda2_ptr,
  T* output_ptr);

extern template
void compute_batch_box_cox__avx2_fma<float>(
  std::size_t N,
  std::size_t D,
  std::size_t block_size,
  const float* self_data,
  const float* __restrict lambda1_data,
  const float* __restrict lambda2_data,
  float* output_data);

extern template
void compute_batch_box_cox__avx2_fma<double>(
  std::size_t N,
  std::size_t D,
  std::size_t block_size,
  const double* self_data,
  const double* __restrict lambda1_data,
  const double* __restrict lambda2_data,
  double* output_data);
} // namespace detail
#endif

template <typename T>
void compute_batch_box_cox(
    std::size_t N,
    std::size_t D,
    std::size_t block_size,
    const T* data,
    const T* lambda1_data,
    const T* lambda2_data,
    T* output_data) {
#ifdef CAFFE2_PERF_WITH_AVX2
  AVX2_FMA_DO(
    details::compute_batch_box_cox,
    N,
    D,
    block_size,
    data,
    lambda1_data,
    lambda2_data,
    output_data);
#endif
  BoxCoxNaive<T>(N, D, data, lambda1_data, lambda2_data, output_data);
}

template void compute_batch_box_cox<float>(
  std::size_t N,
  std::size_t D,
  std::size_t block_size,
  const float* data,
  const float* lambda1_data,
  const float* lambda2_data,
  float* output_data);

template void compute_batch_box_cox<double>(
  std::size_t N,
  std::size_t D,
  std::size_t block_size,
  const double* data,
  const double* lambda1_data,
  const double* lambda2_data,
  double* output_data);

} // namespace caffe2
