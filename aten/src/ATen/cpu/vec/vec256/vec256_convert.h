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
struct VecConvert<Half, 1, float, 1> {
  static inline VectorizedN<Half, 1> apply(const VectorizedN<float, 1>& src) {
    VectorizedN<Half, 1> result;
    result[0] = _mm256_castsi128_si256(cvtfp32_fp16(src[0]));
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
inline Vectorized<int32_t> convert_to_int_of_same_size<float>(
    const Vectorized<float>& src);

template <>
struct VecConvert<int64_t, 2, float, 1> {
  static inline VectorizedN<int64_t, 2> apply(
      const VectorizedN<float, 1>& src) {
    at::vec::VectorizedN<int64_t, 2> result;
    auto int32_vec = at::vec::convert_to_int_of_same_size(src[0]);
    result[0] = _mm256_cvtepi32_epi64(_mm256_castsi256_si128(int32_vec));
    result[1] = _mm256_cvtepi32_epi64(_mm256_extracti128_si256(int32_vec, 1));
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

template <typename dst_t>
struct VecConvert<
  dst_t,
  1,
  float,
  1,
  typename std::enable_if_t<
    std::is_same_v<dst_t, unsigned char> || std::is_same_v<dst_t, signed char>,
    void>> {
  static inline VectorizedN<dst_t, 1> apply(
      const VectorizedN<float, 1>& src) {
    return convert_float_to_int8<dst_t>(src[0]);
  }
};

template <typename src_t>
struct VecConvert<
  float,
  1,
  src_t,
  1,
  typename std::enable_if_t<
    std::is_same_v<src_t, unsigned char> || std::is_same_v<src_t, signed char>,
    void>> {
  static inline VectorizedN<float, 1> apply(
      const VectorizedN<src_t, 1>& src) {
    return convert_int8_to_float<src_t>(src[0]);
  }
};

template <typename dst_t>
struct VecConvert<
    dst_t,
    1,
    int64_t,
    2,
    typename std::enable_if<
        std::is_same_v<dst_t, int8_t> ||
        std::is_same_v<dst_t, uint8_t>>::type> {
  static inline VectorizedN<dst_t, 1> apply(
      const VectorizedN<int64_t, 2>& src) {
    return VecConvert<dst_t, 1, int32_t, 1>::apply(
        VecConvert<int32_t, 1, int64_t, 2>::apply(src));
  }
};

#endif

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
