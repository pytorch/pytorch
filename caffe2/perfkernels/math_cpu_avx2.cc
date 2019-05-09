// Implements the math functions for CPU.
// The implementation in this file allows us to route the underlying numerical
// computation library to different compiler options (-mno-avx2 or -mavx2).

#include <immintrin.h>
#include <cmath>
#include <cstdint>

using std::uint64_t;
using std::uint8_t;

namespace caffe2 {

namespace math {

static constexpr double QEPSILON = 1e-8;

void quantize_and_compress__avx2(
    const float* input_data,
    uint8_t* output_data,
    uint64_t input_size,
    uint64_t bitwidth,
    bool random,
    const float* random_buffer) {
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
  __m256i permute_mask_v =
      _mm256_set_epi32(0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x04, 0x00);

  uint64_t data_per_byte = 8 / bitwidth;
  uint64_t tail = input_size % data_per_byte;
  tail = tail ? data_per_byte - tail : 0;
  uint64_t segment_size = (input_size + data_per_byte - 1) / data_per_byte;

  // basic info
  float minimum_element = INFINITY, maximum_element = -INFINITY;
  for (auto i = 0; i < input_size; ++i) {
    minimum_element =
        (input_data[i] < minimum_element) ? input_data[i] : minimum_element;
    maximum_element =
        (input_data[i] > maximum_element) ? input_data[i] : maximum_element;
  }
  output_data[0] = bitwidth;
  output_data[1] = tail;
  reinterpret_cast<float*>(output_data + 2)[0] = minimum_element;
  reinterpret_cast<float*>(output_data + 2)[1] = maximum_element;

  float gap = (maximum_element - minimum_element) / ((1 << bitwidth) - 1.0f);
  float gap_inverse = 1. / (gap + QEPSILON);
  uint8_t max_q = (1 << bitwidth) - 1;
  uint64_t bit_start = 0;
  if (random) {
    for (int start = 0; start < input_size; start += segment_size) {
      uint64_t stride = start + segment_size <= input_size ? segment_size
                                                           : input_size - start;
      int i = 0;
      constexpr int VLEN = 8;
      for (; i < stride / VLEN * VLEN; i += VLEN) {
        __m256 r_v = _mm256_loadu_ps(&random_buffer[start + i]);
        __m256 fval_v = _mm256_loadu_ps(input_data + start + i);
        __m256 thetimes_v = _mm256_mul_ps(
            _mm256_sub_ps(fval_v, _mm256_set1_ps(minimum_element)),
            _mm256_set1_ps(gap_inverse));
        __m256 rounded_v = _mm256_floor_ps(_mm256_add_ps(thetimes_v, r_v));
        rounded_v = _mm256_max_ps(
            _mm256_setzero_ps(),
            _mm256_min_ps(_mm256_set1_ps(max_q), rounded_v));
        __m256i qval_v = _mm256_cvtps_epi32(rounded_v);
        __m256i orval_v = _mm256_cvtepu8_epi32(_mm_lddqu_si128(
            reinterpret_cast<const __m128i*>(output_data + 10 + i)));
        orval_v =
            _mm256_or_si256(orval_v, _mm256_slli_epi32(qval_v, bit_start));
        orval_v = _mm256_shuffle_epi8(orval_v, shuffle_mask_v);
        orval_v = _mm256_permutevar8x32_epi32(orval_v, permute_mask_v);
        *reinterpret_cast<int64_t*>(output_data + 10 + i) =
            _mm256_extract_epi64(orval_v, 0);
      }
      for (; i < stride; ++i) {
        float fval = input_data[start + i];
        float thetimes = (fval - minimum_element) * gap_inverse;
        float rounded = floor(thetimes + random_buffer[start + i]);
        rounded = rounded < static_cast<float>(max_q)
            ? rounded
            : static_cast<float>(max_q);
        rounded = rounded > 0.0f ? rounded : 0.0f;
        uint8_t qval = rounded;

        uint8_t orval = output_data[10 + i];
        output_data[10 + i] = orval | static_cast<uint8_t>(qval << bit_start);
      }
      bit_start += bitwidth;
    }
  } else {
    // !random
    for (int start = 0; start < input_size; start += segment_size) {
      uint64_t stride = start + segment_size <= input_size ? segment_size
                                                           : input_size - start;
      int i = 0;
      constexpr int VLEN = 8;
      for (; i < stride / VLEN * VLEN; i += VLEN) {
        __m256 fval_v = _mm256_loadu_ps(input_data + start + i);
        __m256 thetimes_v = _mm256_mul_ps(
            _mm256_sub_ps(fval_v, _mm256_set1_ps(minimum_element)),
            _mm256_set1_ps(gap_inverse));
        thetimes_v = _mm256_max_ps(
            _mm256_setzero_ps(),
            _mm256_min_ps(_mm256_set1_ps(max_q), thetimes_v));
        __m256i qval_v = _mm256_cvtps_epi32(_mm256_round_ps(
            thetimes_v, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
        __m256i orval_v = _mm256_cvtepu8_epi32(_mm_lddqu_si128(
            reinterpret_cast<const __m128i*>(output_data + 10 + i)));
        orval_v =
            _mm256_or_si256(orval_v, _mm256_slli_epi32(qval_v, bit_start));
        orval_v = _mm256_shuffle_epi8(orval_v, shuffle_mask_v);
        orval_v = _mm256_permutevar8x32_epi32(orval_v, permute_mask_v);
        *reinterpret_cast<int64_t*>(output_data + 10 + i) =
            _mm256_extract_epi64(orval_v, 0);
      }
      for (; i < stride; ++i) {
        float fval = input_data[start + i];
        float thetimes = (fval - minimum_element) * gap_inverse;
        thetimes = thetimes < static_cast<float>(max_q)
            ? thetimes
            : static_cast<float>(max_q);
        thetimes = thetimes > 0.0f ? thetimes : 0.0f;
        uint8_t qval = nearbyint(thetimes);

        uint8_t orval = output_data[10 + i];
        output_data[10 + i] = orval | static_cast<uint8_t>(qval << bit_start);
      }
      bit_start += bitwidth;
    }
  } // !random
}

void decompress_and_dequantize__avx2(
    const uint8_t* input_data,
    float* output_data,
    uint64_t input_size) {
  // basic info
  const float minimum_element =
      reinterpret_cast<const float*>(input_data + 2)[0];
  const float maximum_element =
      reinterpret_cast<const float*>(input_data + 2)[1];
  const uint64_t bitwidth = input_data[0];
  const float gap =
      (maximum_element - minimum_element) / ((1 << bitwidth) - 1.f) +
      QEPSILON; // for exact recovering

  const uint64_t tail = input_data[1];

  const uint64_t output_size = (input_size - 10) * (8 / bitwidth) - tail;
  // decoding
  uint64_t bit_start = 0;
  const uint64_t segment_size = input_size - 10;
  for (int start = 0; start < output_size; start += segment_size) {
    uint64_t stride = start + segment_size <= output_size ? segment_size
                                                          : output_size - start;
    uint8_t mask = (1 << bitwidth) - 1;
    int i = 0;
    // Can process 8 elements at a time because we need to expand uint8_t
    // to int32_t to use epi32 vector instructions.
    constexpr int VLEN = 8;
    for (; i < stride / VLEN * VLEN; i += VLEN) {
      __m128i in_v = _mm_lddqu_si128(
          reinterpret_cast<const __m128i*>(input_data + 10 + i));
      __m256i out_epi32_v = _mm256_and_si256(
          _mm256_srli_epi32(_mm256_cvtepu8_epi32(in_v), bit_start),
          _mm256_set1_epi32(mask));
      __m256 out_v = _mm256_fmadd_ps(
          _mm256_cvtepi32_ps(out_epi32_v),
          _mm256_set1_ps(gap),
          _mm256_set1_ps(minimum_element));
      _mm256_storeu_ps(output_data + start + i, out_v);
    }
    for (; i < stride; ++i) {
      output_data[start + i] =
          ((input_data[10 + i] >> bit_start) & mask) * gap + minimum_element;
    }
    bit_start += bitwidth;
  }
}

} // namespace math
} // namespace caffe2
