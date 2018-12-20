#include <algorithm>
#include <cmath>

#include <immintrin.h>

namespace dnnlowp {

namespace internal {

float L2MinimizationKernelAVX2(
    int precision,
    float* bins,
    int nbins,
    float bin_width,
    float dst_bin_width,
    int start_bin) {
  float norm = 0;
  constexpr int VLEN = 8;
  float norm_delta_default = dst_bin_width * dst_bin_width * dst_bin_width / 12;

  __m256i identity_v = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
  __m256 bin_width_v = _mm256_set1_ps(bin_width);
  __m256 bin_width_inverse_v = _mm256_set1_ps(1.0f / bin_width);
  __m256 dst_bin_width_v = _mm256_set1_ps(dst_bin_width);
  __m256 dst_bin_width_inverse_v = _mm256_set1_ps(1.0f / dst_bin_width);
  __m256 norm_v = _mm256_setzero_ps();

  int src_bin = 0;
  for (; src_bin < nbins / VLEN * VLEN; src_bin += VLEN) {
    // distances from the beginning of first dst_bin to the beginning and
    // end of src_bin
    __m256i src_bin_v =
        _mm256_add_epi32(_mm256_set1_epi32(src_bin), identity_v);
    __m256 src_bin_begin_v = _mm256_mul_ps(
        _mm256_cvtepi32_ps(
            _mm256_sub_epi32(src_bin_v, _mm256_set1_epi32(start_bin))),
        bin_width_v);
    __m256 src_bin_end_v = _mm256_add_ps(src_bin_begin_v, bin_width_v);

    // which dst_bins the beginning and end of src_bin belong to?
    __m256i dst_bin_of_begin_v = _mm256_cvtps_epi32(_mm256_max_ps(
        _mm256_setzero_ps(),
        _mm256_min_ps(
            _mm256_floor_ps(
                _mm256_mul_ps(src_bin_begin_v, dst_bin_width_inverse_v)),
            _mm256_set1_ps((1 << precision) - 1.0f))));
    __m256i dst_bin_of_end_v = _mm256_cvtps_epi32(_mm256_max_ps(
        _mm256_setzero_ps(),
        _mm256_min_ps(
            _mm256_floor_ps(
                _mm256_mul_ps(src_bin_end_v, dst_bin_width_inverse_v)),
            _mm256_set1_ps((1 << precision) - 1.0f))));

    __m256 dst_bin_of_begin_center_v = _mm256_fmadd_ps(
        _mm256_cvtepi32_ps(dst_bin_of_begin_v),
        dst_bin_width_v,
        _mm256_set1_ps(dst_bin_width / 2));
    // Using sizeof(float) instead of 4 generates compilation error in dbg mode.
    __m256 density_v = _mm256_mul_ps(
        _mm256_i32gather_ps(bins, src_bin_v, 4), bin_width_inverse_v);
    __m256 delta_begin_v =
        _mm256_sub_ps(src_bin_begin_v, dst_bin_of_begin_center_v);
    __m256 norm_delta_v = _mm256_mul_ps(
        _mm256_mul_ps(
            _mm256_mul_ps(delta_begin_v, delta_begin_v), delta_begin_v),
        _mm256_set1_ps(-1.0f / 3));
    __m256i mask_v = _mm256_cmpeq_epi32(dst_bin_of_begin_v, dst_bin_of_end_v);

    __m256 delta_end0_v =
        _mm256_sub_ps(src_bin_end_v, dst_bin_of_begin_center_v);

    __m256 dst_bin_of_end_center_v = _mm256_fmadd_ps(
        _mm256_cvtepi32_ps(dst_bin_of_end_v),
        dst_bin_width_v,
        _mm256_set1_ps(dst_bin_width / 2));
    __m256 delta_end1_v = _mm256_sub_ps(src_bin_end_v, dst_bin_of_end_center_v);
    __m256 delta_end_v = _mm256_blendv_ps(
        delta_end1_v, delta_end0_v, _mm256_castsi256_ps(mask_v));
    norm_delta_v = _mm256_fmadd_ps(
        _mm256_mul_ps(_mm256_mul_ps(delta_end_v, delta_end_v), delta_end_v),
        _mm256_set1_ps(1.0f / 3),
        norm_delta_v);

    norm_delta_v = _mm256_fmadd_ps(
        _mm256_cvtepi32_ps(
            _mm256_sub_epi32(dst_bin_of_end_v, dst_bin_of_begin_v)),
        _mm256_set1_ps(norm_delta_default),
        norm_delta_v);

    norm_v = _mm256_fmadd_ps(density_v, norm_delta_v, norm_v);
  } // src_bin loop vectorized
  float norm_buf[VLEN];
  _mm256_storeu_ps(norm_buf, norm_v);
  for (int i = 0; i < VLEN; ++i) {
    norm += norm_buf[i];
  }

  for (; src_bin < nbins; ++src_bin) {
    // distances from the beginning of first dst_bin to the beginning and
    // end of src_bin
    float src_bin_begin = (src_bin - start_bin) * bin_width;
    float src_bin_end = src_bin_begin + bin_width;

    // which dst_bins the beginning and end of src_bin belong to?
    int dst_bin_of_begin = std::min(
        (1 << precision) - 1.0f,
        std::max(0.0f, floorf(src_bin_begin / dst_bin_width)));
    int dst_bin_of_end = std::min(
        (1 << precision) - 1.0f,
        std::max(0.0f, floorf(src_bin_end / dst_bin_width)));

    float dst_bin_of_begin_center =
        dst_bin_of_begin * dst_bin_width + dst_bin_width / 2;
    float density = bins[src_bin] / bin_width;
    float delta_begin = src_bin_begin - dst_bin_of_begin_center;
    float norm_delta = -(delta_begin * delta_begin * delta_begin) / 3;
    if (dst_bin_of_begin == dst_bin_of_end) {
      // if src_bin is entirely within 1 dst_bin
      float delta_end = src_bin_end - dst_bin_of_begin_center;
      norm_delta += (delta_end * delta_end * delta_end) / 3;
    } else {
      norm_delta += (dst_bin_of_end - dst_bin_of_begin) * norm_delta_default;

      float dst_bin_of_end_center =
          dst_bin_of_end * dst_bin_width + dst_bin_width / 2;
      float delta_end = src_bin_end - dst_bin_of_end_center;
      norm_delta += (delta_end * delta_end * delta_end) / 3;
    }
    norm += density * norm_delta;
  } // src_bin loop remainder

  return norm;
}

} // namespace internal

} // namespace dnnlowp
