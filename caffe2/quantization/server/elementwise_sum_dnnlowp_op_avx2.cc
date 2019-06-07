#include <algorithm>
#include <cmath>
#include <cstdint>

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
  __m256i permute_mask_v =
      _mm256_set_epi32(0x07, 0x03, 0x06, 0x02, 0x05, 0x01, 0x04, 0x00);

  int len_aligned = len / (VLEN * 4) * (VLEN * 4);
  int j = 0;
  for (; j < len_aligned; j += VLEN * 4) {
    // Input is uint8_t but cvtepi8_epi32 assumes the input is int8_t,
    // so we subtract 0x80, cvtepi8_epi32, and then add 0x80
    // x
    __m256 in_v0 = _mm256_cvtepi32_ps(_mm256_add_epi32(
        _mm256_cvtepi8_epi32(_mm_sub_epi8(
            _mm_loadl_epi64(reinterpret_cast<const __m128i*>(input0 + j)),
            _mm_set1_epi8(0x80))),
        _mm256_set1_epi32(0x80)));
    in_v0 = _mm256_fmadd_ps(
        in_v0,
        _mm256_set1_ps(a_scale),
        _mm256_set1_ps(-a_zero_point * a_scale - b_zero_point * b_scale));

    __m256 in_v1 = _mm256_cvtepi32_ps(_mm256_add_epi32(
        _mm256_cvtepi8_epi32(_mm_sub_epi8(
            _mm_loadl_epi64(reinterpret_cast<const __m128i*>(input1 + j)),
            _mm_set1_epi8(0x80))),
        _mm256_set1_epi32(0x80)));
    __m256 acc_v = _mm256_fmadd_ps(in_v1, _mm256_set1_ps(b_scale), in_v0);

    __m256 x_transformed_v = _mm256_fmadd_ps(
        acc_v, _mm256_set1_ps(1.0 / c_scale), _mm256_set1_ps(c_zero_point));

    // y
    in_v0 = _mm256_cvtepi32_ps(_mm256_add_epi32(
        _mm256_cvtepi8_epi32(_mm_sub_epi8(
            _mm_loadl_epi64(
                reinterpret_cast<const __m128i*>(input0 + j + VLEN)),
            _mm_set1_epi8(0x80))),
        _mm256_set1_epi32(0x80)));
    in_v0 = _mm256_fmadd_ps(
        in_v0,
        _mm256_set1_ps(a_scale),
        _mm256_set1_ps(-a_zero_point * a_scale - b_zero_point * b_scale));

    in_v1 = _mm256_cvtepi32_ps(_mm256_add_epi32(
        _mm256_cvtepi8_epi32(_mm_sub_epi8(
            _mm_loadl_epi64(
                reinterpret_cast<const __m128i*>(input1 + j + VLEN)),
            _mm_set1_epi8(0x80))),
        _mm256_set1_epi32(0x80)));
    acc_v = _mm256_fmadd_ps(in_v1, _mm256_set1_ps(b_scale), in_v0);

    __m256 y_transformed_v = _mm256_fmadd_ps(
        acc_v, _mm256_set1_ps(1.0 / c_scale), _mm256_set1_ps(c_zero_point));

    // z
    in_v0 = _mm256_cvtepi32_ps(_mm256_add_epi32(
        _mm256_cvtepi8_epi32(_mm_sub_epi8(
            _mm_loadl_epi64(
                reinterpret_cast<const __m128i*>(input0 + j + 2 * VLEN)),
            _mm_set1_epi8(0x80))),
        _mm256_set1_epi32(0x80)));
    in_v0 = _mm256_fmadd_ps(
        in_v0,
        _mm256_set1_ps(a_scale),
        _mm256_set1_ps(-a_zero_point * a_scale - b_zero_point * b_scale));

    in_v1 = _mm256_cvtepi32_ps(_mm256_add_epi32(
        _mm256_cvtepi8_epi32(_mm_sub_epi8(
            _mm_loadl_epi64(
                reinterpret_cast<const __m128i*>(input1 + j + 2 * VLEN)),
            _mm_set1_epi8(0x80))),
        _mm256_set1_epi32(0x80)));
    acc_v = _mm256_fmadd_ps(in_v1, _mm256_set1_ps(b_scale), in_v0);

    __m256 z_transformed_v = _mm256_fmadd_ps(
        acc_v, _mm256_set1_ps(1.0 / c_scale), _mm256_set1_ps(c_zero_point));

    // w
    in_v0 = _mm256_cvtepi32_ps(_mm256_add_epi32(
        _mm256_cvtepi8_epi32(_mm_sub_epi8(
            _mm_loadl_epi64(
                reinterpret_cast<const __m128i*>(input0 + j + 3 * VLEN)),
            _mm_set1_epi8(0x80))),
        _mm256_set1_epi32(0x80)));
    in_v0 = _mm256_fmadd_ps(
        in_v0,
        _mm256_set1_ps(a_scale),
        _mm256_set1_ps(-a_zero_point * a_scale - b_zero_point * b_scale));

    in_v1 = _mm256_cvtepi32_ps(_mm256_add_epi32(
        _mm256_cvtepi8_epi32(_mm_sub_epi8(
            _mm_loadl_epi64(
                reinterpret_cast<const __m128i*>(input1 + j + 3 * VLEN)),
            _mm_set1_epi8(0x80))),
        _mm256_set1_epi32(0x80)));
    acc_v = _mm256_fmadd_ps(in_v1, _mm256_set1_ps(b_scale), in_v0);

    __m256 w_transformed_v = _mm256_fmadd_ps(
        acc_v, _mm256_set1_ps(1.0 / c_scale), _mm256_set1_ps(c_zero_point));

    // See fbgemm/src/QuantUtilsAvx2.cc requantizeOutputProcessingAvx2 function
    // for more details on this instruction sequence
    __m256i x_rounded_v = _mm256_cvtps_epi32(x_transformed_v);
    __m256i y_rounded_v = _mm256_cvtps_epi32(y_transformed_v);
    __m256i z_rounded_v = _mm256_cvtps_epi32(z_transformed_v);
    __m256i w_rounded_v = _mm256_cvtps_epi32(w_transformed_v);

    __m256i xy_packed_v = _mm256_packs_epi32(x_rounded_v, y_rounded_v);
    __m256i zw_packed_v = _mm256_packs_epi32(z_rounded_v, w_rounded_v);
    __m256i xyzw_packed_v = _mm256_packus_epi16(xy_packed_v, zw_packed_v);
    __m256i xyzw_clamped_v = _mm256_max_epu8(
        ReluFused ? _mm256_set1_epi8(c_zero_point) : _mm256_setzero_si256(),
        _mm256_min_epu8(
            xyzw_packed_v, _mm256_set1_epi8(static_cast<uint8_t>(255))));

    xyzw_clamped_v =
        _mm256_permutevar8x32_epi32(xyzw_clamped_v, permute_mask_v);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(output + j), xyzw_clamped_v);
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
