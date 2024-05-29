#pragma once

#include <cstdint>

namespace caffe2 {

namespace math {

// Returns the quantized and compressed values of floating inputs
// The "fused" representation stores the [bitwidth][tail][min][max]
// with the quantized data in one array. Since we store 8/bitwidth
// quantized data in one byte, the last buckets of some bytes may have
// unused bits. There are totally tail buckets are unused.
// We encode *bitwidth* and *tail* at the beginning,
// following by 32-bit floating data respresenting min and max.
// | bitwidth | tail | min | max | ... int8 data ... |
// |    1B    |  1B  |  4B |  4B | ...output_data....|
// In output_data: the b-th bucket of the i-th byte stores
// the i-th data of the b-th segment of input row

void quantize_and_compress(
    const float* input_data,
    std::uint8_t* output_data,
    std::uint64_t input_size,
    std::uint64_t bitwidth,
    bool random,
    const float* random_buffer);

void decompress_and_dequantize(
    const std::uint8_t* input_data,
    float* output_data,
    std::uint64_t input_size);

} // namespace math
} // namespace caffe2
