#include "./fused_nbit_rowwise_conversion.h"

#include <c10/util/Half.h>
#include <algorithm>
#include <cmath>

#include "common.h"

namespace caffe2 {

void FloatToFused8BitRowwiseQuantized__base(
    const float* input,
    int input_rows,
    int input_columns,
    std::uint8_t* output) {
  constexpr float kEpsilon = 1e-8f;

  int output_columns = input_columns + 2 * sizeof(float);
  for (std::size_t row = 0; row < input_rows; ++row) {
    const float* input_row = input + row * input_columns;
    std::uint8_t* output_row = output + row * output_columns;
    float* output_row_scale_bias =
        reinterpret_cast<float*>(output_row + input_columns);

    float minimum_element =
        *std::min_element(input_row, input_row + input_columns);
    float maximum_element =
        *std::max_element(input_row, input_row + input_columns);
    float range = maximum_element - minimum_element;

    output_row_scale_bias[0] = range / 255.0f;
    output_row_scale_bias[1] = minimum_element;
    const auto inverse_scale = 255.0f / (range + kEpsilon);
    for (std::size_t col = 0; col < input_columns; ++col) {
      output_row[col] =
          std::lrintf((input_row[col] - minimum_element) * inverse_scale);
    }
  }
}

void Fused8BitRowwiseQuantizedToFloat__base(
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

    for (std::size_t col = 0; col < output_columns; ++col) {
      output_row[col] =
          input_row[col] * input_row_scale_bias[0] + input_row_scale_bias[1];
    }
  }
}

decltype(FloatToFused8BitRowwiseQuantized__base)
    FloatToFused8BitRowwiseQuantized__avx2_fma;
void FloatToFused8BitRowwiseQuantized(
    const float* input,
    int input_rows,
    int input_columns,
    std::uint8_t* output) {
  AVX2_FMA_DO(
      FloatToFused8BitRowwiseQuantized,
      input,
      input_rows,
      input_columns,
      output);
  BASE_DO(
      FloatToFused8BitRowwiseQuantized,
      input,
      input_rows,
      input_columns,
      output);
}

decltype(Fused8BitRowwiseQuantizedToFloat__base)
    Fused8BitRowwiseQuantizedToFloat__avx2_fma;
void Fused8BitRowwiseQuantizedToFloat(
    const std::uint8_t* input,
    int input_rows,
    int input_columns,
    float* output) {
  AVX2_FMA_DO(
      Fused8BitRowwiseQuantizedToFloat,
      input,
      input_rows,
      input_columns,
      output);
  BASE_DO(
      Fused8BitRowwiseQuantizedToFloat,
      input,
      input_rows,
      input_columns,
      output);
}

void FloatToFusedNBitRowwiseQuantizedSBHalf__base(
    int bit_rate,
    const float* input,
    int input_rows,
    int input_columns,
    std::uint8_t* output) {
  int num_elem_per_byte = 8 / bit_rate;
  int output_columns =
      (input_columns + num_elem_per_byte - 1) / num_elem_per_byte +
      2 * sizeof(at::Half);
  for (std::size_t row = 0; row < input_rows; ++row) {
    const float* input_row = input + row * input_columns;
    std::uint8_t* output_row = output + row * output_columns;
    at::Half* output_row_scale_bias = reinterpret_cast<at::Half*>(
        output_row +
        (input_columns + num_elem_per_byte - 1) / num_elem_per_byte);

    float minimum_element =
        *std::min_element(input_row, input_row + input_columns);
    float maximum_element =
        *std::max_element(input_row, input_row + input_columns);

    minimum_element = static_cast<at::Half>(minimum_element);
    const float range = maximum_element - minimum_element;

    at::Half scale = range == 0 ? 1.0f : range / ((1 << bit_rate) - 1);
    if (scale == 0) {
      // Corner case handling when maximum_element == minimum_element
      // Any scale would work because X - minimum_element will be 0 for all X
      scale = 1.0f;
    }
    float inverse_scale = 1.0f / scale;
    if (std::isinf(inverse_scale)) {
      scale = 1.0f;
      inverse_scale = 1.0f;
    }

    output_row_scale_bias[0] = scale;
    output_row_scale_bias[1] = minimum_element;
    for (std::size_t col = 0; col < input_columns; ++col) {
      float X = input_row[col];
      std::uint8_t quantized = std::max(
          0,
          std::min<int>(
              std::lrintf((X - minimum_element) * inverse_scale),
              (1 << bit_rate) - 1));
      if (col % num_elem_per_byte == 0) {
        output_row[col / num_elem_per_byte] = quantized;
      } else {
        output_row[col / num_elem_per_byte] |=
            (quantized << ((col % num_elem_per_byte) * bit_rate));
      }
    }
  }
}

void FusedNBitRowwiseQuantizedSBHalfToFloat__base(
    int bit_rate,
    const std::uint8_t* input,
    int input_rows,
    int input_columns,
    float* output) {
  int num_elem_per_byte = 8 / bit_rate;
  int output_columns =
      (input_columns - 2 * sizeof(at::Half)) * num_elem_per_byte;

  for (std::size_t row = 0; row < input_rows; ++row) {
    const std::uint8_t* input_row = input + row * input_columns;
    const at::Half* input_row_scale_bias = reinterpret_cast<const at::Half*>(
        input_row +
        (output_columns + num_elem_per_byte - 1) / num_elem_per_byte);
    float scale = input_row_scale_bias[0];
    float bias = input_row_scale_bias[1];
    float* output_row = output + row * output_columns;

    for (std::size_t col = 0; col < output_columns; ++col) {
      std::uint8_t quantized = input_row[col / num_elem_per_byte];
      quantized >>= (col % num_elem_per_byte) * bit_rate;
      quantized &= (1 << bit_rate) - 1;
      output_row[col] = scale * quantized + bias;
    }
  }
}

decltype(FloatToFusedNBitRowwiseQuantizedSBHalf__base)
    FloatToFusedNBitRowwiseQuantizedSBHalf__avx2_fma;
void FloatToFusedNBitRowwiseQuantizedSBHalf(
    int bit_rate,
    const float* input,
    int input_rows,
    int input_columns,
    std::uint8_t* output) {
  AVX2_FMA_DO(
      FloatToFusedNBitRowwiseQuantizedSBHalf,
      bit_rate,
      input,
      input_rows,
      input_columns,
      output);
  BASE_DO(
      FloatToFusedNBitRowwiseQuantizedSBHalf,
      bit_rate,
      input,
      input_rows,
      input_columns,
      output);
}

decltype(FusedNBitRowwiseQuantizedSBHalfToFloat__base)
    FusedNBitRowwiseQuantizedSBHalfToFloat__avx2_fma;
void FusedNBitRowwiseQuantizedSBHalfToFloat(
    int bit_rate,
    const std::uint8_t* input,
    int input_rows,
    int input_columns,
    float* output) {
  AVX2_FMA_DO(
      FusedNBitRowwiseQuantizedSBHalfToFloat,
      bit_rate,
      input,
      input_rows,
      input_columns,
      output);
  BASE_DO(
      FusedNBitRowwiseQuantizedSBHalfToFloat,
      bit_rate,
      input,
      input_rows,
      input_columns,
      output);
}

} // namespace caffe2
