// Implements the math functions for CPU.
// The implementation in this file allows us to route the underlying numerical
// computation library to different compiler options (-mno-avx2 or -mavx2).

#include <immintrin.h>
#include <cmath>
#include <cstdint>

using std::uint64_t;

namespace caffe2 {

namespace memory {

static inline void float_memory_copy_block64(
    float* input_data,
    float* output_data) {
  __m256 tmp_values0 = _mm256_loadu_ps(&input_data[0]);
  __m256 tmp_values1 = _mm256_loadu_ps(&input_data[8]);
  __m256 tmp_values2 = _mm256_loadu_ps(&input_data[16]);
  __m256 tmp_values3 = _mm256_loadu_ps(&input_data[24]);
  __m256 tmp_values4 = _mm256_loadu_ps(&input_data[32]);
  __m256 tmp_values5 = _mm256_loadu_ps(&input_data[40]);
  __m256 tmp_values6 = _mm256_loadu_ps(&input_data[48]);
  __m256 tmp_values7 = _mm256_loadu_ps(&input_data[56]);
  _mm256_storeu_ps(&output_data[0], tmp_values0);
  _mm256_storeu_ps(&output_data[8], tmp_values1);
  _mm256_storeu_ps(&output_data[16], tmp_values2);
  _mm256_storeu_ps(&output_data[24], tmp_values3);
  _mm256_storeu_ps(&output_data[32], tmp_values4);
  _mm256_storeu_ps(&output_data[40], tmp_values5);
  _mm256_storeu_ps(&output_data[48], tmp_values6);
  _mm256_storeu_ps(&output_data[56], tmp_values7);
}

static inline void float_memory_copy_block128(
    float* input_data,
    float* output_data) {
  __m256 tmp_values0 = _mm256_loadu_ps(&input_data[0]);
  __m256 tmp_values1 = _mm256_loadu_ps(&input_data[8]);
  __m256 tmp_values2 = _mm256_loadu_ps(&input_data[16]);
  __m256 tmp_values3 = _mm256_loadu_ps(&input_data[24]);
  __m256 tmp_values4 = _mm256_loadu_ps(&input_data[32]);
  __m256 tmp_values5 = _mm256_loadu_ps(&input_data[40]);
  __m256 tmp_values6 = _mm256_loadu_ps(&input_data[48]);
  __m256 tmp_values7 = _mm256_loadu_ps(&input_data[56]);
  __m256 tmp_values8 = _mm256_loadu_ps(&input_data[64]);
  __m256 tmp_values9 = _mm256_loadu_ps(&input_data[72]);
  __m256 tmp_values10 = _mm256_loadu_ps(&input_data[80]);
  __m256 tmp_values11 = _mm256_loadu_ps(&input_data[88]);
  __m256 tmp_values12 = _mm256_loadu_ps(&input_data[96]);
  __m256 tmp_values13 = _mm256_loadu_ps(&input_data[104]);
  __m256 tmp_values14 = _mm256_loadu_ps(&input_data[112]);
  __m256 tmp_values15 = _mm256_loadu_ps(&input_data[120]);
  _mm256_storeu_ps(&output_data[0], tmp_values0);
  _mm256_storeu_ps(&output_data[8], tmp_values1);
  _mm256_storeu_ps(&output_data[16], tmp_values2);
  _mm256_storeu_ps(&output_data[24], tmp_values3);
  _mm256_storeu_ps(&output_data[32], tmp_values4);
  _mm256_storeu_ps(&output_data[40], tmp_values5);
  _mm256_storeu_ps(&output_data[48], tmp_values6);
  _mm256_storeu_ps(&output_data[56], tmp_values7);
  _mm256_storeu_ps(&output_data[64], tmp_values8);
  _mm256_storeu_ps(&output_data[72], tmp_values9);
  _mm256_storeu_ps(&output_data[80], tmp_values10);
  _mm256_storeu_ps(&output_data[88], tmp_values11);
  _mm256_storeu_ps(&output_data[96], tmp_values12);
  _mm256_storeu_ps(&output_data[104], tmp_values13);
  _mm256_storeu_ps(&output_data[112], tmp_values14);
  _mm256_storeu_ps(&output_data[120], tmp_values15);
}

void float_memory_region_select_copy__avx2(
    uint64_t one_region_size,
    uint64_t select_start,
    uint64_t select_end,
    float* input_data,
    float* output_data)
{
  if(one_region_size == 128) {
    for(auto s = select_start; s < select_end; s++) {
      auto output_region = output_data + one_region_size * s;
      float_memory_copy_block128((float*)(input_data), (float*)(output_region));
    }
  } else if (one_region_size == 64) {
    for(auto s = select_start; s < select_end; s++) {
      auto output_region = output_data + one_region_size * s;
      float_memory_copy_block64((float*)(input_data), (float*)(output_region));
    }
  } else {
    uint64_t aligned_size = (one_region_size >> 6) << 6;
    int n = one_region_size - aligned_size;
    for(auto s = select_start; s < select_end; s++) {
      auto output_region = output_data + one_region_size * s;
      for (auto d = 0; d < aligned_size; d += 64) {
        float_memory_copy_block64((float*)(input_data + d), (float*)(output_region + d));
      }
      if (n > 0) {
        for (uint64_t dn = aligned_size; dn < one_region_size; dn++) {
          output_region[dn] = input_data[dn];
        }
      }
    }
  }
}

} // namespace math
} // namespace caffe2
