#pragma once

#include <cstddef>
#include <cstdint>

namespace caffe2 {

namespace math {

void quantize_and_compress__avx2(
    const float* input_data,
    uint8_t* output_data,
    std::size_t input_size,
    std::size_t bitwidth,
    bool random,
    const float* random_buffer);

void decompress_and_dequantize__avx2(
    const uint8_t* input_data,
    float* output_data,
    std::size_t input_size);

} // namespace math
} // namespace caffe2
