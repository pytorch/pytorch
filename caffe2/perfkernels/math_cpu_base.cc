// Implements the math functions for CPU.
// The implementation in this file allows us to route the underlying numerical
// computation library to different compiler options (-mno-avx2 or -mavx2).

#include <cfloat>

#include "common.h"
#include "math.h"

namespace caffe2 {

namespace math {

static constexpr double QEPSILON = 1e-8;

void quantize_and_compress__base(
    const float* input_data,
    uint8_t* output_data,
    size_t input_size,
    size_t bitwidth,
    bool random,
    const float* random_buffer) {
  size_t data_per_byte = 8 / bitwidth;
  size_t tail = input_size % data_per_byte;
  tail = tail ? data_per_byte - tail : 0;
  size_t segment_size = (input_size + data_per_byte - 1) / data_per_byte;

  // basic info
  float minimum_element = INFINITY, maximum_element = -INFINITY;
  for (auto i = 0; i < input_size; ++i) {
    minimum_element =
        input_data[i] < minimum_element ? input_data[i] : minimum_element;
    maximum_element =
        input_data[i] > maximum_element ? input_data[i] : maximum_element;
  }
  output_data[0] = bitwidth;
  output_data[1] = tail;
  reinterpret_cast<float*>(output_data + 2)[0] = minimum_element;
  reinterpret_cast<float*>(output_data + 2)[1] = maximum_element;

  float gap = (maximum_element - minimum_element) / ((1 << bitwidth) - 1.0f);
  float gap_inverse = 1. / (gap + QEPSILON);
  uint8_t max_q = (1 << bitwidth) - 1;
  size_t bit_start = 0;
  if (random) {
    for (int start = 0; start < input_size; start += segment_size) {
      size_t stride = start + segment_size <= input_size ? segment_size
                                                         : input_size - start;
      int i = 0;
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
    for (int start = 0; start < input_size; start += segment_size) {
      size_t stride = start + segment_size <= input_size ? segment_size
                                                         : input_size - start;
      int i = 0;
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
  }
}

void quantize_and_compress(
    const float* input_data,
    uint8_t* output_data,
    size_t input_size,
    size_t bitwidth,
    bool random,
    const float* random_buffer) {
  AVX2_DO(
      quantize_and_compress,
      input_data,
      output_data,
      input_size,
      bitwidth,
      random,
      random_buffer);
  BASE_DO(
      quantize_and_compress,
      input_data,
      output_data,
      input_size,
      bitwidth,
      random,
      random_buffer);
}

void decompress_and_dequantize__base(
    const uint8_t* input_data,
    float* output_data,
    size_t input_size) {
  // basic info
  const float minimum_element =
      reinterpret_cast<const float*>(input_data + 2)[0];
  const float maximum_element =
      reinterpret_cast<const float*>(input_data + 2)[1];
  const size_t bitwidth = input_data[0];
  const float gap =
      (maximum_element - minimum_element) / ((1 << bitwidth) - 1.f) +
      QEPSILON; // for exact recovering

  const size_t tail = input_data[1];

  const size_t output_size = (input_size - 10) * (8 / bitwidth) - tail;
  // decoding
  size_t bit_start = 0;
  const size_t segment_size = input_size - 10;
  for (int start = 0; start < output_size; start += segment_size) {
    size_t stride = start + segment_size <= output_size ? segment_size
                                                        : output_size - start;
    uint8_t mask = (1 << bitwidth) - 1;
    int i = 0;
    for (; i < stride; ++i) {
      output_data[start + i] =
          ((input_data[10 + i] >> bit_start) & mask) * gap + minimum_element;
    }
    bit_start += bitwidth;
  }
}

void decompress_and_dequantize(
    const uint8_t* input_data,
    float* output_data,
    size_t input_size) {
  AVX2_DO(decompress_and_dequantize, input_data, output_data, input_size);
  BASE_DO(decompress_and_dequantize, input_data, output_data, input_size);
}

} // namespace math
} // namespace caffe2
