#pragma once

#include <ATen/ATen.h>
#include <ATen/cpu/vec/functional.h>
#include <ATen/cpu/vec/vec.h>
#include <ATen/native/transformers/cpu/utils.h>
#include <utility>

#ifdef _OPENMP
#include <omp.h>

namespace at {
namespace native {

static auto vec_size = vec::Vectorized<float>::size();

inline void _exp_reduce_sum_fusion_kernel(
    float* a,
    const int& size,
    float* out,
    float& val) {
#pragma omp declare reduction(                                    \
        + : vec::Vectorized<float> : omp_out = omp_out += omp_in) \
    initializer(                                                  \
            omp_priv = 0)
  auto vec_max = vec::Vectorized<float>(val);
  float tmp_sum = 0;
  auto vec_tmp_sum = vec::Vectorized<float>(tmp_sum);
  for (long i = 0; i < vec_size * (size / vec_size); i += vec_size) {
    auto tmp0 = vec::Vectorized<float>::loadu(a + i);
    auto tmp1 = tmp0 - vec_max;
    auto tmp2 = tmp1.exp();
    vec_tmp_sum += tmp2;
    _store(out + i, tmp2);
  }
  tmp_sum = vec::vec_reduce_all<float>(
      [](vec::Vectorized<float>& x, vec::Vectorized<float>& y) {
        return x + y;
      },
      vec_tmp_sum);
#pragma omp simd simdlen(8) reduction(+ : tmp_sum)
  for (long i = vec_size * (size / vec_size); i < size; i++) {
    auto tmp0 = a[i];
    auto tmp1 = tmp0 - val;
    auto tmp2 = std::exp(tmp1);
    tmp_sum += tmp2;
    out[i] = tmp2;
  }
  val = tmp_sum;
}

template <typename scalar_t>
inline void _normalization_kernel(
    const float* a,
    const float& sum,
    const int& size,
    scalar_t* out) {
  auto vec_sum = vec::Vectorized<float>(sum);
  for (long i = 0; i < vec_size * (size / vec_size); i += vec_size) {
    auto tmp0 = vec::Vectorized<float>::loadu(a + i);
    auto tmp1 = tmp0 / vec_sum;
    _store(out + i, tmp1);
  }
#pragma omp simd simdlen(8)
  for (long i = vec_size * (size / vec_size); i < size; i++) {
    auto tmp0 = a[i];
    auto tmp1 = tmp0 / sum;
    out[i] = tmp1;
  }
}

inline void _mul_reduce_max_fusion_kernel(
    const float* a,
    const float& scale,
    const int& size,
    float* out,
    float& max) {
#pragma omp declare reduction(                   \
        max : vec::Vectorized<float> : omp_out = \
        vec::maximum(omp_out, omp_in))           \
    initializer(                                 \
            omp_priv = -std::numeric_limits<float>::infinity())
  auto vec_scale = vec::Vectorized<float>(scale);
  float tmp_max = -std::numeric_limits<float>::infinity();
  auto vec_tmp_max = vec::Vectorized<float>(tmp_max);
  for (long i = 0; i < vec_size * (size / vec_size); i += vec_size) {
    auto tmp0 = vec::Vectorized<float>::loadu(a + i);
    auto tmp1 = tmp0 * vec_scale;
    vec_tmp_max = vec::maximum(vec_tmp_max, tmp1);
    _store(out + i, tmp1);
  }
#pragma omp simd simdlen(8) reduction(max : tmp_max)
  for (long i = vec_size * (size / vec_size); i < size; i++) {
    auto tmp0 = a[i];
    auto tmp1 = tmp0 * scale;
    tmp_max = std::max(tmp_max, tmp1);
    out[i] = tmp1;
  }
  max = std::max(
      tmp_max,
      vec::vec_reduce_all<float>(
          [](vec::Vectorized<float>& x, vec::Vectorized<float>& y) {
            return vec::maximum(x, y);
          },
          vec_tmp_max));
}

inline void _init_mha_buffer_kernel(float* max, float* sum, const int& size) {
  float tmp_max = -std::numeric_limits<float>::infinity();
  auto vec_tmp_max = vec::Vectorized<float>(tmp_max);
  float tmp_zero = 0;
  auto vec_tmp_zero = vec::Vectorized<float>(tmp_zero);
  for (long i = 0; i < vec_size * (size / vec_size); i += vec_size) {
    _store(max + i, vec_tmp_max);
    _store(sum + i, vec_tmp_zero);
  }
#pragma omp simd simdlen(8)
  for (long i = vec_size * (size / vec_size); i < size; i++) {
    max[i] = tmp_max;
    sum[i] = tmp_zero;
  }
}

/**
 * This kernel is used to reorder the MHA output
 * with strides.
 * src: MKL GEMM output buffer
 * dst: Final MHA output
 */
template <typename scalar_t>
inline void _reorder_mha_output_kernel(
    float* src,
    scalar_t* dst,
    const int& rows,
    const int& cols,
    const int& dst_stride) {
  for (long i = 0; i < rows; ++i) {
    for (long j = 0; j < vec_size * (cols / vec_size); j += vec_size) {
      auto tmp0 = vec::Vectorized<float>::loadu(src + i * cols + j);
      _store(dst + i * dst_stride + j, tmp0);
    }
#pragma omp simd simdlen(8)
    for (long j = vec_size * (cols / vec_size); j < cols; j++) {
      dst[i * dst_stride + j] = src[i * cols + j];
    }
  }
}

/**
 * This kernel is used to update the MHA output with the latest MAX
 * and SUM values block by block.
 * exp_val: exp(max_old - max_new)
 * In the i th block, the softmax(qk - i th) * v - i th was calculated
 * with the old MAX and SUM values, max_old and sum_old. When moving to
 * the i + 1 th block, since softmax(qk - i + 1 th) will be calculated
 * with the new MAX and SUM values, max_new and sum_new, thus the MHA
 * buffer which stores the summation of blocked softmax(qk) * v should
 * be also updated using max_new and sum_new:
 * a = a * sum_old / sum_new
 * a = a * exp(max_old) / exp(max_new) = a * exp_val
 */
inline void _mha_update_sum_max_kernel(
    const float* a,
    const float& sum_old,
    const float& sum_new,
    const float& exp_val,
    const int& size,
    float* out) {
  float sum_cor = sum_old / sum_new;
  auto vec_sum_cor = vec::Vectorized<float>(sum_cor);
  auto vec_exp = vec::Vectorized<float>(exp_val);
  for (long i = 0; i < vec_size * (size / vec_size); i += vec_size) {
    auto tmp0 = vec::Vectorized<float>::loadu(a + i);
    auto tmp1 = tmp0 * vec_sum_cor;
    auto tmp2 = tmp1 * vec_exp;
    _store(out + i, tmp2);
  }
#pragma omp simd simdlen(8)
  for (long i = vec_size * (size / vec_size); i < size; i++) {
    auto tmp0 = a[i];
    auto tmp1 = tmp0 * sum_cor;
    auto tmp2 = tmp1 * exp_val;
    out[i] = tmp2;
  }
}

} // namespace native
} // namespace at

#endif // _OPENMP
