#include <ATen/ATen.h>
#include <ATen/cpu/vec/functional.h>
#include <ATen/cpu/vec/vec.h>
#include <ATen/native/transformers/cpu/utils.h>
#include <utility>

namespace at {
namespace native {

using fVec = vec::Vectorized<float>;
static auto f_vec_size = fVec::size();

inline void _exp_reduce_sum_fusion_kernel(
    float* a,
    const int& size,
    float* out,
    float& val) {
  vec::map<float>(
    [val](fVec x) { return (x - fVec(val)).exp(); }, out, a, size);
  val = vec::reduce_all<float>(
    [](fVec& x, fVec& y) { return x + y; }, out, size);
}

template <typename scalar_t>
inline void _normalization_kernel(
    const float* a,
    const float& sum,
    const int& size,
    scalar_t* out) {
  auto vec_sum = fVec(sum);
  int64_t i = 0;
  for (i = 0; i < f_vec_size * (size / f_vec_size); i += f_vec_size) {
    auto tmp0 = fVec::loadu(a + i);
    auto tmp1 = tmp0 / vec_sum;
    _store(out + i, tmp1);
  }
  if (size - i > 0) {
    auto tmp0 = fVec::loadu(a + i, size - i);
    auto tmp1 = tmp0 / vec_sum;
    _store(out + i, tmp1, size - i);
  }
}

inline void _mul_reduce_max_fusion_kernel(
    const float* a,
    const float& scale,
    const int& size,
    float* out,
    float& max) {
  vec::map<float>(
    [scale](fVec x) { return x * fVec(scale); }, out, a, size);
  max = vec::reduce_all<float>(
    [](fVec& x, fVec& y) { return vec::maximum(x, y); }, out, size);
}

inline void _init_mha_buffer_kernel(float* max, float* sum, const int& size) {
  float tmp_max = -std::numeric_limits<float>::infinity();
  auto vec_tmp_max = fVec(tmp_max);
  float tmp_zero = 0;
  auto vec_tmp_zero = fVec(tmp_zero);
  int64_t i = 0;
  for (i = 0; i < f_vec_size * (size / f_vec_size); i += f_vec_size) {
    _store(max + i, vec_tmp_max);
    _store(sum + i, vec_tmp_zero);
  }
  if (size - i > 0) {
    _store(max + i, vec_tmp_max, size - i);
    _store(sum + i, vec_tmp_zero, size - i);
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
  for (int64_t i = 0; i < rows; ++i) {
    int64_t j = 0;
    for (j = 0; j < f_vec_size * (cols / f_vec_size); j += f_vec_size) {
      auto tmp0 = fVec::loadu(src + i * cols + j);
      _store(dst + i * dst_stride + j, tmp0);
    }
    if (cols - j > 0) {
      auto tmp0 = fVec::loadu(src + i * cols + j, cols - j);
      _store(dst + i * dst_stride + j, tmp0, cols - j);
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
  vec::map<float>(
    [sum_cor, exp_val](fVec x)
      { return x * fVec(sum_cor) * fVec(exp_val); },
      out, a, size);
}

} // namespace native
} // namespace at
