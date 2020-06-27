#include "./fused_nbit_rowwise_conversion.h"

#include <c10/util/Half.h>
#include <algorithm>
#include <cmath>
#include <vector>

#include <c10/util/Half.h>
#include "caffe2/core/logging.h"

#ifdef _OPENMP
#include <omp.h>
#endif

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

void FloatToFused4BitRowwiseQuantizedHelper(
    const float* input_row,
    int input_rows,
    int input_columns,
    int BIT_RATE,
    int NUM_ELEM_PER_BYTE,
    bool GREEDY,
    void (*param_search_callback)(
        const float* X,
        int N,
        const int n_bins,
        const float ratio,
        float& Xmin,
        float& Xmax,
        int bit_rate),
    std::uint8_t* output_row) {
  // const float* input_row = input + row * input_columns;
  // std::uint8_t* output_row = output + row * output_columns;
  at::Half* output_row_scale = reinterpret_cast<at::Half*>(
      output_row + (input_columns + NUM_ELEM_PER_BYTE - 1) / NUM_ELEM_PER_BYTE);

  at::Half* output_row_bias = reinterpret_cast<at::Half*>(
      output_row + (input_columns + NUM_ELEM_PER_BYTE - 1) / NUM_ELEM_PER_BYTE +
      sizeof(at::Half));

  float Xmin = *std::min_element(input_row, input_row + input_columns);
  float Xmax = *std::max_element(input_row, input_row + input_columns);

  if (GREEDY) {
    C10_LOG_EVERY_N(INFO, 100) << "Running the GREEDY engine!";
    param_search_callback(
        input_row, input_columns, 200, 0.16, Xmin, Xmax, BIT_RATE);
  }
  // Round Xmin to fp16 to match with dequantization that will use fp16
  // for Xmin.
  Xmin = static_cast<at::Half>(Xmin);
  const float range = Xmax - Xmin;
  // Round scale to fp16 to match with dequantization that will use fp16
  // for scale.
  // Set scale to 1.0f for the corner case of Xmax == Xmin .
  // Any non-zero scale would work because during quantization
  // (X - Xmin) / scale will be 0 for all X unless scale is 0.
  at::Half scale = range == 0 ? 1.0f : range / ((1 << BIT_RATE) - 1);
  if (scale == 0) {
    // Corner case handling when Xmax == Xmin
    // Any scale would work because X - Xmin will be 0 for all X
    scale = 1.0f;
  }

  *output_row_scale = scale;
  *output_row_bias = Xmin;

  for (int col = 0; col < input_columns; ++col) {
    float X = input_row[col];
    std::uint8_t quantized = std::max(
        0, std::min<int>(std::lrintf((X - Xmin) / scale), (1 << BIT_RATE) - 1));
    if (col % NUM_ELEM_PER_BYTE == 0) {
      // LSB
      output_row[col / NUM_ELEM_PER_BYTE] = quantized;
    } else {
      output_row[col / NUM_ELEM_PER_BYTE] |=
          (quantized << ((col % NUM_ELEM_PER_BYTE) * BIT_RATE));
    }
  }
}

void FloatToFused4BitRowwiseQuantized(
    const float* input,
    int input_rows,
    int input_columns,
    std::uint8_t* output) {
  const int BIT_RATE = 4;
  constexpr int NUM_ELEM_PER_BYTE = 8 / BIT_RATE;
  const int output_columns = static_cast<std::int64_t>(
      (input_columns + NUM_ELEM_PER_BYTE - 1) / NUM_ELEM_PER_BYTE +
      2 * sizeof(at::Half));
  for (int row = 0; row < input_rows; ++row) {
    const float* input_row = input + row * input_columns;
    std::uint8_t* output_row = output + row * output_columns;
    FloatToFused4BitRowwiseQuantizedHelper(
        input_row,
        input_rows,
        input_columns,
        BIT_RATE,
        NUM_ELEM_PER_BYTE,
        false,
        nullptr,
        output_row);
  }
}

} // namespace caffe2
