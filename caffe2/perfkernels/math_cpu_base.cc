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
#include "caffe2/perfkernels/common.h"
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

void quantize_and_compress__base(
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

void quantize_and_compress(
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
#ifdef FUSED_ROWWISE_RANDOM_QUANTIZATION_USE_MKL
  AVX2_DO(
      quantize_and_compress,
      input_data,
      output_data,
      input_size,
      bitwidth,
      random,
      vslStream,
      random_buffer);
  BASE_DO(
      quantize_and_compress,
      input_data,
      output_data,
      input_size,
      bitwidth,
      random,
      vslStream,
      random_buffer);
#else
  AVX2_DO(
      quantize_and_compress,
      input_data,
      output_data,
      input_size,
      bitwidth,
      random,
      dis,
      gen);
  BASE_DO(
      quantize_and_compress,
      input_data,
      output_data,
      input_size,
      bitwidth,
      random,
      dis,
      gen);
#endif
}

void decompress_and_dequantize__base(
    const uint8_t* input_data,
    float* output_data,
    size_t input_size) {
  // memory pointers ///
  ConstEigenVectorArrayMap<uint8_t> input_bitwidth_tail(input_data, 2);
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
    for (; i < stride; ++i) {
      output_data[start + i] = ((input_data[10 + i] >> bit_start) & mask);
    }
    bit_start += bitwidth;
  }
  // scaling and biasing
  output_row = output_row * gap + minimum_element;
}

void decompress_and_dequantize(
    const uint8_t* input_data,
    float* output_data,
    size_t input_size) {
  AVX2_DO(decompress_and_dequantize, input_data, output_data, input_size);
  BASE_DO(decompress_and_dequantize, input_data, output_data, input_size);
}

#undef QEPSILON
} // namespace math
} // namespace caffe2
