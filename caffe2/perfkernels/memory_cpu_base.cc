// Implements the math functions for CPU.
// The implementation in this file allows us to route the underlying numerical
// computation library to different compiler options (-mno-avx2 or -mavx2).

#include <cfloat>
#include <cstdint>
#include <immintrin.h>
#include <string.h>

#include "common.h"
#include "memory.h"

using std::uint64_t;

namespace caffe2 {

namespace memory {

void float_memory_region_select_copy__base(
    uint64_t one_region_size,
    uint64_t select_start,
    uint64_t select_end,
    float* input_data,
    float* output_data)
{
  uint64_t aligned_size = (one_region_size >> 6) << 6;
  int n = one_region_size - aligned_size;
  for(auto s = select_start; s < select_end; s++) {
    auto output_region = output_data + one_region_size * s;
    for (auto d = 0; d < aligned_size; d += 64) {
      memcpy((void*)(&input_data[d]), (void*)(&output_region[d]), 64 * sizeof(float));
    }
    if (n > 0) {
      for (uint64_t dn = aligned_size; dn < one_region_size; dn++) {
        output_region[dn] = input_data[dn];
      }
    }
  }
}

decltype(float_memory_region_select_copy__base) float_memory_region_select_copy__avx2;
void float_memory_region_select_copy(
    uint64_t one_region_size,
    uint64_t select_start,
    uint64_t select_end,
    float* input_data,
    float* output_data) {
  AVX2_DO(float_memory_region_select_copy, one_region_size, select_start, select_end, input_data, output_data);
  BASE_DO(float_memory_region_select_copy, one_region_size, select_start, select_end, input_data, output_data);
}

} // namespace memory
} // namespace caffe2
