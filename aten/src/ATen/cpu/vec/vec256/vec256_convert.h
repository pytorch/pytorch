#pragma once

#include <ATen/cpu/vec/functional_bfloat16.h>
#include <ATen/cpu/vec/intrinsics.h>
#include <ATen/cpu/vec/vec_base.h>
#include <ATen/cpu/vec/vec_convert.h>

namespace at::vec {
inline namespace CPU_CAPABILITY {

#if defined(CPU_CAPABILITY_AVX2) && !defined(_MSC_VER)

template <>
struct VecConvert<float, 1, BFloat16, 1> {
  static inline VectorizedN<float, 1> apply(
      const VectorizedN<BFloat16, 1>& src) {
    VectorizedN<float, 1> result;
    __m256 value;
    cvtbf16_fp32(_mm256_castsi256_si128(src[0]), value);
    result[0] = value;
    return result;
  }
};

template <>
struct VecConvert<float, 1, Half, 1> {
  static inline VectorizedN<float, 1> apply(const VectorizedN<Half, 1>& src) {
    VectorizedN<float, 1> result;
    __m256 value;
    cvtfp16_fp32(_mm256_castsi256_si128(src[0]), value);
    result[0] = value;
    return result;
  }
};

template <>
struct VecConvert<BFloat16, 1, float, 1> {
  static inline VectorizedN<BFloat16, 1> apply(
      const VectorizedN<float, 1>& src) {
    VectorizedN<BFloat16, 1> result;
    result[0] = _mm256_castsi128_si256(cvtfp32_bf16(src[0]));
    return result;
  }
};

template <>
struct VecConvert<BFloat16, 1, float, 2> {
  static inline VectorizedN<BFloat16, 1> apply(
      const VectorizedN<float, 2>& src) {
    VectorizedN<BFloat16, 1> result;
    result[0] = convert_float_bfloat16(src[0], src[1]);
    return result;
  }
};

template <>
struct VecConvert<float, 2, BFloat16, 1> {
  static inline VectorizedN<float, 2> apply(
      const VectorizedN<BFloat16, 1>& src) {
    VectorizedN<float, 2> result;
    std::tie(result[0], result[1]) = convert_bfloat16_float(src[0]);
    return result;
  }
};

template <>
struct VecConvert<Half, 1, float, 1> {
  static inline VectorizedN<Half, 1> apply(const VectorizedN<float, 1>& src) {
    VectorizedN<Half, 1> result;
    result[0] = _mm256_castsi128_si256(cvtfp32_fp16(src[0]));
    return result;
  }
};

template <>
struct VecConvert<Half, 1, float, 2> {
  static inline VectorizedN<Half, 1> apply(const VectorizedN<float, 2>& src) {
    VectorizedN<Half, 1> result;
    result[0] = convert_float_half(src[0], src[1]);
    return result;
  }
};

template <>
struct VecConvert<float, 2, Half, 1> {
  static inline VectorizedN<float, 2> apply(const VectorizedN<Half, 1>& src) {
    VectorizedN<float, 2> result;
    std::tie(result[0], result[1]) = convert_half_float(src[0]);
    return result;
  }
};

template <>
inline Vectorized<double> convert_to_fp_of_same_size<double>(
    const Vectorized<int64_t>& src);

template <>
struct VecConvert<float, 1, int64_t, 2> {
  static inline VectorizedN<float, 1> apply(
      const VectorizedN<int64_t, 2>& src) {
    auto low_double = at::vec::convert_to_fp_of_same_size<double>(src[0]);
    auto low = _mm256_cvtpd_ps(low_double);
    auto high_double = at::vec::convert_to_fp_of_same_size<double>(src[1]);
    auto high = _mm256_cvtpd_ps(high_double);
    return Vectorized<float>(
        _mm256_insertf128_ps(_mm256_castps128_ps256(low), high, 1));
  }
};

template <>
struct VecConvert<int64_t, 2, float, 1> {
  static inline VectorizedN<int64_t, 2> apply(
      const VectorizedN<float, 1>& src) {
    // Scalarization is the most reliable way of converting fp to int64 on AVX2.
    // Check: https://stackoverflow.com/questions/41144668
    float buffer[8];
    src.store(buffer);
    at::vec::VectorizedN<int64_t, 2> result;
    result[0] = Vectorized<int64_t>(
      static_cast<int64_t>(buffer[0]),
      static_cast<int64_t>(buffer[1]),
      static_cast<int64_t>(buffer[2]),
      static_cast<int64_t>(buffer[3]));
    result[1] = Vectorized<int64_t>(
      static_cast<int64_t>(buffer[4]),
      static_cast<int64_t>(buffer[5]),
      static_cast<int64_t>(buffer[6]),
      static_cast<int64_t>(buffer[7]));
    return result;
  }
};

template <>
struct VecConvert<int32_t, 1, int64_t, 2> {
  static inline VectorizedN<int32_t, 1> apply(
      const VectorizedN<int64_t, 2>& src) {
    auto low = _mm256_shuffle_epi32(src[0], _MM_SHUFFLE(2, 0, 2, 0));
    auto high = _mm256_shuffle_epi32(src[1], _MM_SHUFFLE(2, 0, 2, 0));
    auto low_perm = _mm256_permute4x64_epi64(low, _MM_SHUFFLE(3, 1, 2, 0));
    auto high_perm = _mm256_permute4x64_epi64(high, _MM_SHUFFLE(3, 1, 2, 0));
    return Vectorized<int32_t>(_mm256_blend_epi32(low_perm, high_perm, 0xF0));
  }
};

template <>
struct VecConvert<int64_t, 2, int32_t, 1> {
  static inline VectorizedN<int64_t, 2> apply(
      const VectorizedN<int32_t, 1>& src) {
    at::vec::VectorizedN<int64_t, 2> result;
    result[0] = _mm256_cvtepi32_epi64(_mm256_castsi256_si128(src[0]));
    result[1] = _mm256_cvtepi32_epi64(_mm256_extracti128_si256(src[0], 1));
    return result;
  }
};

template <>
struct VecConvert<int32_t, 1, int8_t, 1> {
  static inline VectorizedN<int32_t, 1> apply(
      const VectorizedN<int8_t, 1>& src) {
    auto src128 = _mm256_castsi256_si128(src[0]);
    return Vectorized<int32_t>(_mm256_cvtepi8_epi32(src128));
  }
};

template <>
struct VecConvert<int32_t, 1, uint8_t, 1> {
  static inline VectorizedN<int32_t, 1> apply(
      const VectorizedN<uint8_t, 1>& src) {
    auto src128 = _mm256_castsi256_si128(src[0]);
    return Vectorized<int32_t>(_mm256_cvtepu8_epi32(src128));
  }
};


template <>
struct VecConvert<int32_t, 1, float, 1> {
  static inline VectorizedN<int32_t, 1> apply(
      const VectorizedN<float, 1>& src) {
    return  Vectorized<int32_t>(_mm256_cvttps_epi32(src[0]));
  }
};

template <>
struct VecConvert<float, 1, int32_t, 1> {
  static inline VectorizedN<float, 1> apply(
      const VectorizedN<int32_t, 1>& src) {
    return  Vectorized<float>(_mm256_cvtepi32_ps(src[0]));
  }
};

template <>
struct VecConvert<int16_t, 1, uint8_t, 1> {
  static inline VectorizedN<int16_t, 1> apply(
      const VectorizedN<uint8_t, 1>& src) {
    auto src128 = _mm256_castsi256_si128(src[0]);
    return Vectorized<int16_t>(_mm256_cvtepu8_epi16(src128));
  }
};

template <typename dst_t, typename src_t>
struct VecConvert<
    dst_t,
    1,
    src_t,
    1,
    typename std::enable_if_t<
        (is_reduced_floating_point_v<dst_t> && is_8bit_integer_v<src_t>) ||
            (is_reduced_floating_point_v<src_t> && is_8bit_integer_v<dst_t>),
        void>> {
  static inline VectorizedN<dst_t, 1> apply(const VectorizedN<src_t, 1>& src) {
    VectorizedN<float, 2> tmp_fp32 = VecConvert<float, 2, src_t, 1>::apply(src);
    return VecConvert<dst_t, 1, float, 2>::apply(tmp_fp32);
  }
};

template <typename dst_t>
struct VecConvert<
    dst_t,
    1,
    float,
    2,
    typename std::enable_if_t<is_8bit_integer_v<dst_t>,
        void>> {
  static inline VectorizedN<dst_t, 1> apply(const VectorizedN<float, 2>& src) {
    at::vec::Vectorized<dst_t> vec1 = convert_float_to_int8<dst_t>(src[0]);
    at::vec::Vectorized<dst_t> vec2 = convert_float_to_int8<dst_t>(src[1]);
    __m128 lane2 = _mm256_castps256_ps128(_mm256_castsi256_ps(vec2));
    __m256 combined = _mm256_insertf128_ps(_mm256_castsi256_ps(vec1), lane2, 1);
    // Shuffle [191:128] bit from combined in to [127:64] bit of result
    __m256i result = _mm256_permute4x64_epi64(_mm256_castps_si256(combined), 0b11011000);
    return at::vec::Vectorized<dst_t>(result);
  }
};

template <typename dst_t>
struct VecConvert<
    dst_t,
    1,
    float,
    1,
    typename std::enable_if_t<is_8bit_integer_v<dst_t>,
        void>> {
  static inline VectorizedN<dst_t, 1> apply(const VectorizedN<float, 1>& src) {
    return convert_float_to_int8<dst_t>(src[0]);
  }
};

template <typename src_t>
struct VecConvert<
    float,
    2,
    src_t,
    1,
    typename std::enable_if_t<is_8bit_integer_v<src_t>,
        void>> {
  static inline VectorizedN<float, 2> apply(const VectorizedN<src_t, 1>& src) {
    // Shuffle [127:64] bit from src[0] in to [191:128] bit of shuffled
    __m256i shuffled = _mm256_permute4x64_epi64(src[0], 0b11011000);
    __m256i src2 = _mm256_castsi128_si256(
      _mm_castps_si128(
        _mm256_extractf128_ps(_mm256_castsi256_ps(shuffled), 1) // Extract the second 128-bit lane
      )
    );
    return VectorizedN<float, 2>(convert_int8_to_float<src_t>(src[0]), convert_int8_to_float<src_t>(src2));
  }
};

template <typename dst_t>
struct VecConvert<
    dst_t,
    1,
    int64_t,
    2,
    std::enable_if_t<
        std::is_same_v<dst_t, int8_t> ||
        std::is_same_v<dst_t, uint8_t>>> {
  static inline VectorizedN<dst_t, 1> apply(
      const VectorizedN<int64_t, 2>& src) {
    return VecConvert<dst_t, 1, int32_t, 1>::apply(
        VecConvert<int32_t, 1, int64_t, 2>::apply(src));
  }
};

#endif /* defined(CPU_CAPABILITY_AVX2) && !defined(_MSC_VER) */


#if (defined(CPU_CAPABILITY_AVX2) && !defined(_MSC_VER))
template <typename src_t>
struct VecConvert<
    float,
    1,
    src_t,
    1,
    typename std::enable_if_t<is_8bit_integer_v<src_t>,
        void>> {
  static inline VectorizedN<float, 1> apply(const VectorizedN<src_t, 1>& src) {
    return convert_int8_to_float<src_t>(src[0]);
  }
};
#endif

#if defined(CPU_CAPABILITY_SVE256) && defined(__ARM_FEATURE_BF16)

template <>
struct VecConvert<float, 1, BFloat16, 1> {
  static inline VectorizedN<float, 1> apply(
      const VectorizedN<BFloat16, 1>& src) {
    VectorizedN<float, 1> res;
    // Load 16-bit unsigned integers from src into an SVE vector
    svuint16_t u16x4 = svld1_u16(svptrue_b16(), reinterpret_cast<const uint16_t*>(&src[0]));
    // Zero-extend to 32-bit SVE does not have direct vmovl_u16 equivalent.
    vls_uint32_t u32x4 = svreinterpret_u32_u16(svzip1_u16(svdup_n_u16(0), u16x4));
    // Reinterpret as float32
    vls_float32_t f32x4 = svreinterpret_f32_u32(u32x4);
    res[0] = Vectorized<float>(f32x4);
    return res;
  }
};

template <>
struct VecConvert<float, 2, BFloat16, 1> {
  static inline VectorizedN<float, 2> apply(
      const VectorizedN<BFloat16, 1>& src) {
    VectorizedN<float, 2> res;
    std::tie(res[0], res[1]) = convert_bfloat16_float(src[0]);
    return res;
  }
};

template <>
struct VecConvert<BFloat16, 1, float, 2> {
  static inline VectorizedN<BFloat16, 1> apply(
      const VectorizedN<float, 2>& src) {
    VectorizedN<BFloat16, 1> res;
    res[0] = convert_float_bfloat16(src[0], src[1]);
    return res;
  }
};

#endif // defined(CPU_CAPABILITY_SVE256) && defined(__ARM_FEATURE_BF16)

template <typename src_t>
struct VecConvert<
    float,
    1,
    src_t,
    1,
    typename std::enable_if_t<is_reduced_floating_point_v<src_t>, void>> {
  static inline VectorizedN<float, 1> apply(const VectorizedN<src_t, 1>& src) {
    auto [res_vec1, res_vec2] = convert_to_float<src_t>(src[0]);
    return res_vec1;
  }
};

template <typename dst_t>
struct VecConvert<
    dst_t,
    1,
    float,
    1,
    typename std::enable_if_t<is_reduced_floating_point_v<dst_t>, void>> {
  static inline VectorizedN<dst_t, 1> apply(const VectorizedN<float, 1>& src) {
    return convert_from_float<dst_t>(src[0], src[0]);
  }
};

} // namespace CPU_CAPABILITY
} // namespace at::vec
