#include "fused_8bit_rowwise_conversion.h"

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

void FloatToFused8BitRowwiseQuantizedSBHalf__base(
    const float* input,
    int input_rows,
    int input_columns,
    std::uint8_t* output) {
  int output_columns = input_columns + 2 * sizeof(at::Half);
  for (std::size_t row = 0; row < input_rows; ++row) {
    const float* input_row = input + row * input_columns;
    std::uint8_t* output_row = output + row * output_columns;
    at::Half* output_row_scale_bias =
        reinterpret_cast<at::Half*>(output_row + input_columns);

    float Xmin = *std::min_element(input_row, input_row + input_columns);
    float Xmax = *std::max_element(input_row, input_row + input_columns);

    Xmin = static_cast<at::Half>(Xmin);
    const float range = Xmax - Xmin;

    at::Half scale = range == 0 ? 1.0f : range / 255.0;
    if (scale == 0) {
      // Corner case handling when Xmax == Xmin
      // Any scale would work because X - Xmin will be 0 for all X
      scale = 1.0f;
    }

    output_row_scale_bias[0] = scale;
    output_row_scale_bias[1] = Xmin;
    for (std::size_t col = 0; col < input_columns; ++col) {
      float X = input_row[col];
      std::uint8_t quantized =
          std::max(0, std::min<int>(std::lrintf((X - Xmin) / scale), 255));
      output_row[col] = quantized;
    }
  }
}

void Fused8BitRowwiseQuantizedSBHalfToFloat__base(
    const std::uint8_t* input,
    int input_rows,
    int input_columns,
    float* output) {
  int output_columns = input_columns - 2 * sizeof(at::Half);

  for (std::size_t row = 0; row < input_rows; ++row) {
    const std::uint8_t* input_row = input + row * input_columns;
    const at::Half* input_row_scale_bias =
        reinterpret_cast<const at::Half*>(input_row + output_columns);
    float* output_row = output + row * output_columns;

    for (std::size_t col = 0; col < output_columns; ++col) {
      output_row[col] = (int)input_row[col] * input_row_scale_bias[0] +
          input_row_scale_bias[1];
    }
  }
}

decltype(FloatToFused8BitRowwiseQuantizedSBHalf__base)
    FloatToFused8BitRowwiseQuantizedSBHalf__avx2_fma;
void FloatToFused8BitRowwiseQuantizedSBHalf(
    const float* input,
    int input_rows,
    int input_columns,
    std::uint8_t* output) {
  // TODO: Add AVX2 version
  BASE_DO(
      FloatToFused8BitRowwiseQuantizedSBHalf,
      input,
      input_rows,
      input_columns,
      output);
}

decltype(Fused8BitRowwiseQuantizedSBHalfToFloat__base)
    Fused8BitRowwiseQuantizedSBHalfToFloat__avx2_fma;
void Fused8BitRowwiseQuantizedSBHalfToFloat(
    const std::uint8_t* input,
    int input_rows,
    int input_columns,
    float* output) {
  // TODO: Add AVX2 version
  BASE_DO(
      Fused8BitRowwiseQuantizedSBHalfToFloat,
      input,
      input_rows,
      input_columns,
      output);
}

} // namespace caffe2
