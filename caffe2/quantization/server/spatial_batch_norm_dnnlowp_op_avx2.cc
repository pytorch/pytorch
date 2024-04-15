#include <cstdint>
#include <limits>

#include <immintrin.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <fbgemm/QuantUtils.h>

namespace caffe2 {

namespace internal {

template <typename T>
void SpatialBNNHWCAVX2(
    const int N,
    const int C,
    const int HxW,
    const int in_zero_point,
    const int out_zero_point,
    const T* X,
    const float* alpha,
    const float* beta,
    T* Y,
    bool relu_fused);

template <bool ReluFused>
void SpatialBNNHWCAVX2_uint8(
    const int N,
    const int C,
    const int HxW,
    const int in_zero_point,
    const int out_zero_point,
    const uint8_t* X,
    const float* alpha,
    const float* beta,
    uint8_t* Y) {
  constexpr int kVLen = 8;
  const int outer_size = N * HxW;

  const __m256i min_v = _mm256_set1_epi32(std::numeric_limits<uint8_t>::min());
  const __m256i max_v = _mm256_set1_epi32(std::numeric_limits<uint8_t>::max());
  const __m256i shuffle_mask_v = _mm256_set_epi8(
      // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
      0xff,
      // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
      0xff,
      // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
      0xff,
      // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
      0xff,
      // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
      0xff,
      // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
      0xff,
      // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
      0xff,
      // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
      0xff,
      // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
      0xff,
      // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
      0xff,
      // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
      0xff,
      // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
      0xff,
      0x0c,
      0x08,
      0x04,
      0x00,
      // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
      0xff,
      // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
      0xff,
      // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
      0xff,
      // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
      0xff,
      // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
      0xff,
      // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
      0xff,
      // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
      0xff,
      // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
      0xff,
      // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
      0xff,
      // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
      0xff,
      // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
      0xff,
      // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
      0xff,
      0x0c,
      0x08,
      0x04,
      0x00);
  const __m256i permute_mask_v =
      _mm256_set_epi32(0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x04, 0x00);
  const __m256i in_zero_point_v = _mm256_set1_epi32(in_zero_point);
  const __m256i out_zero_point_v = _mm256_set1_epi32(out_zero_point);
#ifdef _OPENMP
#pragma omp parallel for
#endif
  for (int i = 0; i < outer_size; ++i) {
    int n = C / kVLen * kVLen;
    int r = C % kVLen;
    const uint8_t* X_ptr = X + i * C;
    uint8_t* Y_ptr = Y + i * C;
    for (int j = 0; j < n; j += kVLen) {
      const __m256i cur_v = _mm256_cvtepu8_epi32(
          _mm_loadl_epi64(reinterpret_cast<const __m128i*>(X_ptr + j)));
      const __m256 cur_v_float =
          _mm256_cvtepi32_ps(_mm256_sub_epi32(cur_v, in_zero_point_v));
      const __m256 alpha_v = _mm256_loadu_ps(alpha + j);
      const __m256 beta_v = _mm256_loadu_ps(beta + j);
      const __m256 result_float_v =
          _mm256_fmadd_ps(alpha_v, cur_v_float, beta_v);
      const __m256i result_rounded_v = _mm256_cvtps_epi32(result_float_v);
      __m256i result_v = _mm256_add_epi32(result_rounded_v, out_zero_point_v);
      if (ReluFused) {
        result_v = _mm256_max_epi32(result_v, out_zero_point_v);
      }
      __m256i clipped_v =
          _mm256_max_epi32(min_v, _mm256_min_epi32(max_v, result_v));
      clipped_v = _mm256_shuffle_epi8(clipped_v, shuffle_mask_v);
      clipped_v = _mm256_permutevar8x32_epi32(clipped_v, permute_mask_v);
      *reinterpret_cast<int64_t*>(Y_ptr + j) =
          _mm256_extract_epi64(clipped_v, 0);
    }
    for (int j = 0; j < r; ++j) {
      long quantized_down = out_zero_point +
          // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
          std::lrintf(alpha[n + j] * (X_ptr[n + j] - in_zero_point) +
                      beta[n + j]);
      if (ReluFused) { // static if
        quantized_down = std::max<long>(quantized_down, out_zero_point);
      }
      Y_ptr[n + j] = fbgemm::clamp<long, uint8_t>(quantized_down, 8);
    }
  }
}

template <>
void SpatialBNNHWCAVX2<uint8_t>(
    const int N,
    const int C,
    const int HxW,
    const int in_zero_point,
    const int out_zero_point,
    const uint8_t* X,
    const float* alpha,
    const float* beta,
    uint8_t* Y,
    bool relu_fused) {
  if (relu_fused) {
    SpatialBNNHWCAVX2_uint8<true>(
        N, C, HxW, in_zero_point, out_zero_point, X, alpha, beta, Y);
  } else {
    SpatialBNNHWCAVX2_uint8<false>(
        N, C, HxW, in_zero_point, out_zero_point, X, alpha, beta, Y);
  }
}

} // namespace internal

} // namespace caffe2
