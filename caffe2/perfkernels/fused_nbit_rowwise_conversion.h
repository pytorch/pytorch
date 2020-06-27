#pragma once

#include <c10/util/Half.h>
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

void FloatToFused4BitRowwiseQuantized(
    const float* input,
    int input_rows,
    int input_columns,
    std::uint8_t* output);

/**
 * Row-wise quantization with fp16 scale and bias
 *
 * @param bit_rate can be 2, 4, or 8
 */
void FloatToFusedNBitRowwiseQuantizedSBHalf(
    int bit_rate,
    const float* input,
    int input_rows,
    int input_columns,
    std::uint8_t* output);

void FusedNBitRowwiseQuantizedSBHalfToFloat(
    int bit_rate,
    const std::uint8_t* input,
    int input_rows,
    int input_columns,
    float* output);

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
    std::uint8_t* output_row);

} // namespace caffe2
