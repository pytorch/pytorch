#pragma once

#include <cstdint>

namespace caffe2 {

void FloatToFused8BitRowwiseQuantized(
    const float* input,
    int input_rows,
    int input_columns,
    std::uint8_t* output);

void Fused8BitRowwiseQuantizedToFloat(
    const std::uint8_t* input,
    int input_rows,
    int input_columns,
    float* output);

void FloatToFused8BitRowwiseQuantizedSBHalf(
    const float* input,
    int input_rows,
    int input_columns,
    std::uint8_t* output);

void Fused8BitRowwiseQuantizedSBHalfToFloat(
    const std::uint8_t* input,
    int input_rows,
    int input_columns,
    float* output);

} // namespace caffe2
