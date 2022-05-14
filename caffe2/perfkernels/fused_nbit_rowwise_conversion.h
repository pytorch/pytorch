#pragma once

#include <cstddef>
#include <cstdint>

namespace caffe2 {

void FloatToFused8BitRowwiseQuantized(
    const float* input,
    size_t input_rows,
    int input_columns,
    std::uint8_t* output);

void Fused8BitRowwiseQuantizedToFloat(
    const std::uint8_t* input,
    size_t input_rows,
    int input_columns,
    float* output);

/**
 * Row-wise quantization with fp16 scale and bias
 *
 * @param bit_rate can be 2, 4, or 8
 */
void FloatToFusedNBitRowwiseQuantizedSBHalf(
    int bit_rate,
    const float* input,
    size_t input_rows,
    int input_columns,
    std::uint8_t* output);

void FusedNBitRowwiseQuantizedSBHalfToFloat(
    int bit_rate,
    const std::uint8_t* input,
    size_t input_rows,
    int input_columns,
    float* output);

} // namespace caffe2
