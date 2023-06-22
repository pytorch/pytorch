#include "./fused_nbit_rowwise_conversion.h"

#include <c10/util/Half.h>
#include <algorithm>
#include <cmath>

#include "common.h"

#ifdef USE_FBGEMM
#include "fbgemm/QuantUtils.h"
#endif

namespace caffe2 {

static void FloatToFused8BitRowwiseQuantized__base(
    const float* input,
    size_t input_rows,
    int input_columns,
    std::uint8_t* output) {
  constexpr float kEpsilon = 1e-8f;

  // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
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
    for (std::size_t col = 0; col < static_cast<size_t>(input_columns); ++col) {
      output_row[col] =
          std::lrintf((input_row[col] - minimum_element) * inverse_scale);
    }
  }
}

static void Fused8BitRowwiseQuantizedToFloat__base(
    const std::uint8_t* input,
    size_t input_rows,
    int input_columns,
    float* output) {
  // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
  int output_columns = input_columns - 2 * sizeof(float);

  for (std::size_t row = 0; row < input_rows; ++row) {
    const std::uint8_t* input_row = input + row * input_columns;
    const float* input_row_scale_bias =
        reinterpret_cast<const float*>(input_row + output_columns);
    float* output_row = output + row * output_columns;

    for (std::size_t col = 0; col < static_cast<std::size_t>(output_columns); ++col) {
      output_row[col] =
          // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
          input_row[col] * input_row_scale_bias[0] + input_row_scale_bias[1];
    }
  }
}

void FloatToFused8BitRowwiseQuantized(
    const float* input,
    size_t input_rows,
    int input_columns,
    std::uint8_t* output) {
#ifdef USE_FBGEMM
  fbgemm::FloatOrHalfToFused8BitRowwiseQuantizedSBFloat<float>(
      input, input_rows, input_columns, output);
#else
  FloatToFused8BitRowwiseQuantized__base(
      input, input_rows, input_columns, output);
#endif
}

void Fused8BitRowwiseQuantizedToFloat(
    const std::uint8_t* input,
    size_t input_rows,
    int input_columns,
    float* output) {
#ifdef USE_FBGEMM
  fbgemm::Fused8BitRowwiseQuantizedSBFloatToFloatOrHalf<float>(
      input, input_rows, input_columns, output);
#else
  Fused8BitRowwiseQuantizedToFloat__base(
      input, input_rows, input_columns, output);
#endif
}

static void FloatToFusedNBitRowwiseQuantizedSBHalf__base(
    int bit_rate,
    const float* input,
    size_t input_rows,
    int input_columns,
    std::uint8_t* output) {
  int num_elem_per_byte = 8 / bit_rate;
  int output_columns =
      // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
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

    // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
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
    for (std::size_t col = 0; col < static_cast<size_t>(input_columns); ++col) {
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

static void FusedNBitRowwiseQuantizedSBHalfToFloat__base(
    int bit_rate,
    const std::uint8_t* input,
    size_t input_rows,
    int input_columns,
    float* output) {
  int num_elem_per_byte = 8 / bit_rate;
  int output_columns =
      // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
      (input_columns - 2 * sizeof(at::Half)) * num_elem_per_byte;

  for (std::size_t row = 0; row < static_cast<size_t>(input_rows); ++row) {
    const std::uint8_t* input_row = input + row * input_columns;
    const at::Half* input_row_scale_bias = reinterpret_cast<const at::Half*>(
        input_row +
        (output_columns + num_elem_per_byte - 1) / num_elem_per_byte);
    float scale = input_row_scale_bias[0];
    float bias = input_row_scale_bias[1];
    float* output_row = output + row * output_columns;

    for (std::size_t col = 0; col < static_cast<std::size_t>(output_columns); ++col) {
      std::uint8_t quantized = input_row[col / num_elem_per_byte];
      quantized >>= (col % num_elem_per_byte) * bit_rate;
      quantized &= (1 << bit_rate) - 1;
      // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
      output_row[col] = scale * quantized + bias;
    }
  }
}

void FloatToFusedNBitRowwiseQuantizedSBHalf(
    int bit_rate,
    const float* input,
    size_t input_rows,
    int input_columns,
    std::uint8_t* output) {
#ifdef USE_FBGEMM
  fbgemm::FloatOrHalfToFusedNBitRowwiseQuantizedSBHalf<float>(
      bit_rate, input, input_rows, input_columns, output);
#else
  FloatToFusedNBitRowwiseQuantizedSBHalf__base(
      bit_rate, input, input_rows, input_columns, output);
#endif
}

void FusedNBitRowwiseQuantizedSBHalfToFloat(
    int bit_rate,
    const std::uint8_t* input,
    size_t input_rows,
    int input_columns,
    float* output) {
#ifdef USE_FBGEMM
  fbgemm::FusedNBitRowwiseQuantizedSBHalfToFloatOrHalf<float>(
      bit_rate, input, input_rows, input_columns, output);
#else
  FusedNBitRowwiseQuantizedSBHalfToFloat__base(
      bit_rate, input, input_rows, input_columns, output);
#endif
}

} // namespace caffe2
