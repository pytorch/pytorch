// Implements the math functions for CPU.
// The implementation in this file allows us to route the underlying numerical
// computation library to different compiler options (-mno-avx2 or -mavx2).

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <functional>
#include <limits>
#include <numeric>
#include <random>
#include <tuple>
#include <unordered_set>
#include <vector>

#include "caffe2/core/context.h"
#include "caffe2/perfkernels/math.h"
#include "caffe2/utils/cpu_neon.h"
#include "caffe2/utils/cpuid.h"
#include "caffe2/utils/eigen_utils.h"
#include "caffe2/utils/math.h"

#include "Eigen/Core"
#include "Eigen/Dense"

#ifdef CAFFE2_USE_MKL
#include <mkl.h>
#endif // CAFFE2_USE_MKL

#ifdef CAFFE2_USE_HPTT
#include <hptt.h>
#endif // CAFFE2_USE_HPTT

#if defined(_MSC_VER)
#include <process.h>
#endif

namespace caffe2 {

namespace math {
#define QEPSILON 1e-8

void quantize_and_compress__avx2(
    const float* input_data,
    uint8_t* output_data,
    size_t input_size,
    size_t bitwidth,
    bool random,
#ifdef FUSED_ROWWISE_RANDOM_QUANTIZATION_USE_MKL
    VSLStreamStatePtr& vslStream,
    std::vector<float>& random_buffer
#else
    std::unique_ptr<std::uniform_real_distribution<float>>& dis,
    std::minstd_rand& gen
#endif
) {
  CAFFE_ENFORCE(
      bitwidth == 1 || bitwidth == 2 || bitwidth == 4 || bitwidth == 8,
      "Unsupported bitwidth");

#ifdef __AVX2__
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
#endif // __AVX2__

  // memory pointers
  ConstEigenVectorArrayMap<float> input_row(input_data, input_size);
  uint8_t* output_row = output_data;
  EigenVectorArrayMap<uint8_t> output_bitwidth_tail(output_row, 2);
  EigenVectorArrayMap<float> output_row_min_max(
      reinterpret_cast<float*>(output_row + 2), 2);

  size_t data_per_byte = 8 / bitwidth;
  size_t tail = input_size % data_per_byte;
  tail = tail ? data_per_byte - tail : 0;
  size_t segment_size = (input_size + data_per_byte - 1) / data_per_byte;

  // basic info
  const float minimum_element = input_row.minCoeff();
  const float maximum_element = input_row.maxCoeff();
  output_bitwidth_tail(0) = bitwidth;
  output_bitwidth_tail(1) = tail;
  output_row_min_max(0) = minimum_element;
  output_row_min_max(1) = maximum_element;

  float gap = (maximum_element - minimum_element) / ((1 << bitwidth) - 1.0f);
  float gap_inverse = 1. / (gap + QEPSILON);
  uint8_t max_q = (1 << bitwidth) - 1;
  size_t bit_start = 0;
  if (random) {
#ifdef FUSED_ROWWISE_RANDOM_QUANTIZATION_USE_MKL
    int status = vsRngUniform(
        VSL_RNG_METHOD_UNIFORM_STD,
        vslStream,
        input_size,
        random_buffer.data(),
        0.0f,
        1.0f);
    if (status != VSL_ERROR_OK) {
      LOG(WARNING) << "vsRngUniform returns " << status;
    }
#endif
    for (int start = 0; start < input_size; start += segment_size) {
      size_t stride = start + segment_size <= input_size ? segment_size
                                                         : input_size - start;
      int i = 0;
#ifdef __AVX2__
      constexpr int VLEN = 8;
      for (; i < stride / VLEN * VLEN; i += VLEN) {
#ifdef FUSED_ROWWISE_RANDOM_QUANTIZATION_USE_MKL
        __m256 r_v = _mm256_loadu_ps(&random_buffer[start + i]);
#else
        float random_buffer[VLEN];
        for (int j = 0; j < VLEN; ++j) {
          random_buffer[j] = (*dis)(gen);
        }
        __m256 r_v = _mm256_loadu_ps(random_buffer);
#endif
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
            reinterpret_cast<const __m128i*>(output_row + 10 + i)));
        orval_v =
            _mm256_or_si256(orval_v, _mm256_slli_epi32(qval_v, bit_start));
        orval_v = _mm256_shuffle_epi8(orval_v, shuffle_mask_v);
        orval_v = _mm256_permutevar8x32_epi32(orval_v, permute_mask_v);
        *reinterpret_cast<int64_t*>(output_row + 10 + i) =
            _mm256_extract_epi64(orval_v, 0);
      }
#endif // __AVX2__
      for (; i < stride; ++i) {
        float fval = input_data[start + i];
        float thetimes = (fval - minimum_element) * gap_inverse;
#ifdef FUSED_ROWWISE_RANDOM_QUANTIZATION_USE_MKL
        float rounded = floor(thetimes + random_buffer[start + i]);
#else
        float rounded = floor(thetimes + (*dis)(gen));
#endif
        rounded = std::max(0.0f, std::min(static_cast<float>(max_q), rounded));
        uint8_t qval = rounded;

        uint8_t orval = output_row[10 + i];
        output_row[10 + i] = orval | static_cast<uint8_t>(qval << bit_start);
      }
      bit_start += bitwidth;
    }
  } else {
    for (int start = 0; start < input_size; start += segment_size) {
      size_t stride = start + segment_size <= input_size ? segment_size
                                                         : input_size - start;
      int i = 0;
#ifdef __AVX2__
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
            reinterpret_cast<const __m128i*>(output_row + 10 + i)));
        orval_v =
            _mm256_or_si256(orval_v, _mm256_slli_epi32(qval_v, bit_start));
        orval_v = _mm256_shuffle_epi8(orval_v, shuffle_mask_v);
        orval_v = _mm256_permutevar8x32_epi32(orval_v, permute_mask_v);
        *reinterpret_cast<int64_t*>(output_row + 10 + i) =
            _mm256_extract_epi64(orval_v, 0);
      }
#endif // __AVX2__
      for (; i < stride; ++i) {
        float fval = input_data[start + i];
        float thetimes = (fval - minimum_element) * gap_inverse;
        thetimes =
            std::max(0.0f, std::min(static_cast<float>(max_q), thetimes));
        uint8_t qval = nearbyint(thetimes);

        uint8_t orval = output_row[10 + i];
        output_row[10 + i] = orval | static_cast<uint8_t>(qval << bit_start);
      }
      bit_start += bitwidth;
    }
  }
}

void decompress_and_dequantize__avx2(
    const uint8_t* input_data,
    float* output_data,
    size_t input_size) {
  ConstEigenVectorArrayMap<float> input_row_min_max(
      reinterpret_cast<const float*>(input_data + 2), 2);

  // basic info
  const float minimum_element = input_row_min_max(0);
  const float maximum_element = input_row_min_max(1);
  const size_t bitwidth = input_data[0];
  const float gap =
      (maximum_element - minimum_element) / ((1 << bitwidth) - 1.f) +
      QEPSILON; // for exact recovering

  CAFFE_ENFORCE(
      bitwidth == 1 || bitwidth == 2 || bitwidth == 4 || bitwidth == 8,
      "Unsupported bitwidth");
  const size_t tail = input_data[1];

  const size_t output_size = (input_size - 10) * (8 / bitwidth) - tail;
  EigenVectorArrayMap<float> output_row(output_data, output_size);
  // decoding
  size_t bit_start = 0;
  const size_t segment_size = input_size - 10;
  for (int start = 0; start < output_size; start += segment_size) {
    size_t stride = start + segment_size <= output_size ? segment_size
                                                        : output_size - start;
    uint8_t mask = (1 << bitwidth) - 1;
    int i = 0;
#ifdef __AVX2__
    // Can process 8 elements at a time because we need to expand uint8_t
    // to int32_t to use epi32 vector instructions.
    constexpr int VLEN = 8;
    for (; i < stride / VLEN * VLEN; i += VLEN) {
      __m128i in_v = _mm_lddqu_si128(
          reinterpret_cast<const __m128i*>(input_data + 10 + i));
      __m256i out_epi32_v = _mm256_and_si256(
          _mm256_srli_epi32(_mm256_cvtepu8_epi32(in_v), bit_start),
          _mm256_set1_epi32(mask));
      _mm256_storeu_ps(
          output_data + start + i, _mm256_cvtepi32_ps(out_epi32_v));
    }
#endif
    for (; i < stride; ++i) {
      output_data[start + i] = ((input_data[10 + i] >> bit_start) & mask);
    }
    bit_start += bitwidth;
  }
  // scaling and biasing
  output_row = output_row * gap + minimum_element;
}

#undef QEPSILON
} // namespace math
} // namespace caffe2
