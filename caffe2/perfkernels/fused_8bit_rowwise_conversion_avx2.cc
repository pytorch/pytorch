#include "fused_8bit_rowwise_conversion.h"

#include <immintrin.h>
#include <algorithm>
#include <cfloat>
#include <cmath>

namespace caffe2 {

constexpr int VLEN = 8;

void FloatToFused8BitRowwiseQuantized__avx2_fma(
    const float* input,
    int input_rows,
    int input_columns,
    std::uint8_t* output) {
  constexpr float kEpsilon = 1e-8f;

  __m256i permute_mask1_v =
      _mm256_set_epi32(0x07, 0x03, 0x06, 0x02, 0x05, 0x01, 0x04, 0x00);
  __m256i shuffle_mask_v = _mm256_set_epi8(
      0xff,
      0xff,
      0xff,
      0xff,
      0xff,
      0xff,
      0xff,
      0xff,
      0xff,
      0xff,
      0xff,
      0xff,
      0x0c,
      0x08,
      0x04,
      0x00,
      0xff,
      0xff,
      0xff,
      0xff,
      0xff,
      0xff,
      0xff,
      0xff,
      0xff,
      0xff,
      0xff,
      0xff,
      0x0c,
      0x08,
      0x04,
      0x00);
  __m256i permute_mask2_v =
      _mm256_set_epi32(0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x04, 0x00);

  int output_columns = input_columns + 2 * sizeof(float);
  for (std::size_t row = 0; row < input_rows; ++row) {
    const float* input_row = input + row * input_columns;
    std::uint8_t* output_row = output + row * output_columns;
    float* output_row_scale_bias =
        reinterpret_cast<float*>(output_row + input_columns);

    float minimum_element = FLT_MAX;
    float maximum_element = -FLT_MAX;
    __m256 min_v = _mm256_set1_ps(minimum_element);
    __m256 max_v = _mm256_set1_ps(maximum_element);
    std::size_t col;
    for (col = 0; col < input_columns / VLEN * VLEN; col += VLEN) {
      __m256 in_v = _mm256_loadu_ps(input_row + col);
      min_v = _mm256_min_ps(min_v, in_v);
      max_v = _mm256_max_ps(max_v, in_v);
    }
    alignas(64) float min_buf[VLEN], max_buf[VLEN];
    _mm256_store_ps(min_buf, min_v);
    _mm256_store_ps(max_buf, max_v);
    for (int i = 0; i < VLEN; ++i) {
      minimum_element = std::min(minimum_element, min_buf[i]);
      maximum_element = std::max(maximum_element, max_buf[i]);
    }
    for (; col < input_columns; ++col) {
      minimum_element = std::min(minimum_element, input_row[col]);
      maximum_element = std::max(maximum_element, input_row[col]);
    }

    float range = maximum_element - minimum_element;

    output_row_scale_bias[0] = range / 255.0f;
    output_row_scale_bias[1] = minimum_element;
    const auto inverse_scale = 255.0f / (range + kEpsilon);
    min_v = _mm256_set1_ps(minimum_element);
    __m256 inverse_scale_v = _mm256_set1_ps(inverse_scale);

    for (col = 0; col < input_columns / (4 * VLEN) * (4 * VLEN);
         col += 4 * VLEN) {
      __m256i x_rounded_v = _mm256_cvtps_epi32(_mm256_mul_ps(
          _mm256_sub_ps(_mm256_loadu_ps(input_row + col), min_v),
          inverse_scale_v));
      __m256i y_rounded_v = _mm256_cvtps_epi32(_mm256_mul_ps(
          _mm256_sub_ps(_mm256_loadu_ps(input_row + col + VLEN), min_v),
          inverse_scale_v));
      __m256i z_rounded_v = _mm256_cvtps_epi32(_mm256_mul_ps(
          _mm256_sub_ps(_mm256_loadu_ps(input_row + col + 2 * VLEN), min_v),
          inverse_scale_v));
      __m256i w_rounded_v = _mm256_cvtps_epi32(_mm256_mul_ps(
          _mm256_sub_ps(_mm256_loadu_ps(input_row + col + 3 * VLEN), min_v),
          inverse_scale_v));

      // An instruction sequence to save 32 32-bit integers as 8-bit integers
      __m256i xy_packed_v = _mm256_packs_epi32(x_rounded_v, y_rounded_v);
      __m256i zw_packed_v = _mm256_packs_epi32(z_rounded_v, w_rounded_v);
      __m256i xyzw_packed_v = _mm256_packus_epi16(xy_packed_v, zw_packed_v);
      xyzw_packed_v =
          _mm256_permutevar8x32_epi32(xyzw_packed_v, permute_mask1_v);
      _mm256_storeu_si256(
          reinterpret_cast<__m256i*>(output_row + col), xyzw_packed_v);
    }
    for (; col < input_columns / VLEN * VLEN; col += VLEN) {
      __m256i rounded_v = _mm256_cvtps_epi32(_mm256_mul_ps(
          _mm256_sub_ps(_mm256_loadu_ps(input_row + col), min_v),
          inverse_scale_v));

      // An instruction sequence to save 8 32-bit integers as 8-bit integers
      rounded_v = _mm256_shuffle_epi8(rounded_v, shuffle_mask_v);
      rounded_v = _mm256_permutevar8x32_epi32(rounded_v, permute_mask2_v);
      _mm_storel_epi64(
          reinterpret_cast<__m128i*>(output_row + col),
          _mm256_castsi256_si128(rounded_v));
    }
    for (; col < input_columns; ++col) {
      output_row[col] =
          std::lrintf((input_row[col] - minimum_element) * inverse_scale);
    }
  }
}

void Fused8BitRowwiseQuantizedToFloat__avx2_fma(
    const std::uint8_t* input,
    int input_rows,
    int input_columns,
    float* output) {
  int output_columns = input_columns - 2 * sizeof(float);

  for (std::size_t row = 0; row < input_rows; ++row) {
    const std::uint8_t* input_row = input + row * input_columns;
    const float* input_row_scale_bias =
        reinterpret_cast<const float*>(input_row + output_columns);
    float* output_row = output + row * output_columns;

    __m256 scale_v = _mm256_set1_ps(input_row_scale_bias[0]);
    __m256 bias_v = _mm256_set1_ps(input_row_scale_bias[1]);

    std::size_t col;
    for (col = 0; col < output_columns / VLEN * VLEN; col += VLEN) {
      __m256 in_v = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(
          _mm_loadl_epi64(reinterpret_cast<const __m128i*>(input_row + col))));
      _mm256_storeu_ps(
          output_row + col,
          _mm256_add_ps(_mm256_mul_ps(in_v, scale_v), bias_v));
    }

    for (; col < output_columns; ++col) {
      output_row[col] =
          input_row[col] * input_row_scale_bias[0] + input_row_scale_bias[1];
    }
  }
}

} // namespace caffe2
