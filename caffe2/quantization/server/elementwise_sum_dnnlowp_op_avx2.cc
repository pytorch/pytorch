#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

#include <immintrin.h>

namespace caffe2 {

namespace internal {

using namespace std;

constexpr int VLEN = 8;

template <typename T, bool ReluFused>
void ElementWiseSumAVX2(
    const T* input0,
    const T* input1,
    T* output,
    int len,
    float a_scale,
    int32_t a_zero_point,
    float b_scale,
    int32_t b_zero_point,
    float c_scale,
    int32_t c_zero_point) {
  // TODO: this intrinsic code is replicated in dnnlowp.cc,
  // fbgemm_i8i8_acc32.cc, conv_dnnlowp_op.cc, and here.
  // We need to somehow refactor this.
  __m256 min_v = _mm256_set1_ps(numeric_limits<uint8_t>::min());
  __m256 max_v = _mm256_set1_ps(numeric_limits<uint8_t>::max());

  __m256i shuffle_mask_v = _mm256_set_epi8(
      0xff,
      0xff,
      0xff,
      0xff,
      0xff,
      0xff,
      0xff,
      0xff,
      0xff,
      0xff,
      0xff,
      0xff,
      0x0c,
      0x08,
      0x04,
      0x00,
      0xff,
      0xff,
      0xff,
      0xff,
      0xff,
      0xff,
      0xff,
      0xff,
      0xff,
      0xff,
      0xff,
      0xff,
      0x0c,
      0x08,
      0x04,
      0x00);
  __m256i permute_mask_v =
      _mm256_set_epi32(0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x04, 0x00);

  int len_aligned = len / VLEN * VLEN;
  int j = 0;
  for (; j < len_aligned; j += VLEN) {
    // Input is uint8_t but cvtepi8_epi32 assumes the input is int8_t,
    // so we subtract 0x80, cvtepi8_epi32, and then add 0x80
    __m256 in_v = _mm256_cvtepi32_ps(_mm256_add_epi32(
        _mm256_cvtepi8_epi32(_mm_sub_epi8(
            _mm_loadl_epi64(reinterpret_cast<const __m128i*>(input0 + j)),
            _mm_set1_epi8(0x80))),
        _mm256_set1_epi32(0x80)));
    in_v = _mm256_fmadd_ps(
        in_v, _mm256_set1_ps(a_scale), _mm256_set1_ps(-a_zero_point * a_scale));
    __m256 acc_v = in_v;

    in_v = _mm256_cvtepi32_ps(_mm256_add_epi32(
        _mm256_cvtepi8_epi32(_mm_sub_epi8(
            _mm_loadl_epi64(reinterpret_cast<const __m128i*>(input1 + j)),
            _mm_set1_epi8(0x80))),
        _mm256_set1_epi32(0x80)));
    in_v = _mm256_fmadd_ps(
        in_v, _mm256_set1_ps(b_scale), _mm256_set1_ps(-b_zero_point * b_scale));
    acc_v = _mm256_add_ps(acc_v, in_v);

    __m256 transformed_v = _mm256_fmadd_ps(
        acc_v, _mm256_set1_ps(1.0 / c_scale), _mm256_set1_ps(c_zero_point));
    __m256 clipped_v = _mm256_min_ps(
        _mm256_max_ps(
            transformed_v, ReluFused ? _mm256_set1_ps(c_zero_point) : min_v),
        max_v);
    __m256i rounded_v = _mm256_cvtps_epi32(clipped_v);
    rounded_v = _mm256_shuffle_epi8(rounded_v, shuffle_mask_v);
    rounded_v = _mm256_permutevar8x32_epi32(rounded_v, permute_mask_v);
    *reinterpret_cast<int64_t*>(output + j) =
        _mm256_extract_epi64(rounded_v, 0);
  }
  for (; j < len; ++j) {
    float acc = 0;
    acc += (input0[j] - a_zero_point) * a_scale;
    acc += (input1[j] - b_zero_point) * b_scale;
    float transformed_val = c_zero_point + acc / c_scale;
    output[j] = std::max(
        ReluFused ? c_zero_point : 0.0f,
        std::min(255.0f, nearbyint(transformed_val)));
  }
}

template void ElementWiseSumAVX2<uint8_t, false>(
    const uint8_t* input0,
    const uint8_t* input1,
    uint8_t* output,
    int len,
    float a_scale,
    int32_t a_zero_point,
    float b_scale,
    int32_t b_zero_point,
    float c_scale,
    int32_t c_zero_point);

template void ElementWiseSumAVX2<uint8_t, true>(
    const uint8_t* input0,
    const uint8_t* input1,
    uint8_t* output,
    int len,
    float a_scale,
    int32_t a_zero_point,
    float b_scale,
    int32_t b_zero_point,
    float c_scale,
    int32_t c_zero_point);

template void ElementWiseSumAVX2<uint16_t, false>(
    const uint16_t* input0,
    const uint16_t* input1,
    uint16_t* output,
    int len,
    float a_scale,
    int32_t a_zero_point,
    float b_scale,
    int32_t b_zero_point,
    float c_scale,
    int32_t c_zero_point);

template void ElementWiseSumAVX2<uint16_t, true>(
    const uint16_t* input0,
    const uint16_t* input1,
    uint16_t* output,
    int len,
    float a_scale,
    int32_t a_zero_point,
    float b_scale,
    int32_t b_zero_point,
    float c_scale,
    int32_t c_zero_point);

} // namespace internal

} // namespace caffe2
