#pragma once

#include <cstdint>

using std::uint64_t;

namespace caffe2 {

namespace memory {

void float_memory_region_select_copy(
    uint64_t one_region_size,
    uint64_t select_start,
    uint64_t select_end,
    float* input_data,
    float* output_data);

} // namespace memory
} // namespace caffe2
