#pragma once

#include <ATen/cpu/vec/intrinsics.h>
#include <ATen/cpu/vec/vec_base.h>
#include <ATen/cpu/vec/vec_mask.h>

namespace at::vec {
inline namespace CPU_CAPABILITY {

#if defined(CPU_CAPABILITY_AVX512)

template <typename mask_t>
struct VecMaskLoad<float, 1, mask_t, 1> {
  static inline VectorizedN<float, 1> apply(
      const float* ptr,
      const VecMask<mask_t, 1>& vec_mask) {
    at::vec::Vectorized<float> zero_vec(0);
    auto all_ones = _mm512_set1_epi32(0xFFFFFFFF);
    auto mmask = _mm512_cmp_epi32_mask(
        vec_mask.template cast<int, 1>()[0], all_ones, _MM_CMPINT_EQ);
    return Vectorized<float>(_mm512_mask_loadu_ps(zero_vec, mmask, ptr));
  }
};

template <typename data_t, typename mask_t>
struct VecMaskLoad<
    data_t,
    1,
    mask_t,
    1,
    typename std::enable_if<
        std::is_same_v<data_t, BFloat16> ||
        std::is_same_v<data_t, Half>>::type> {
  static inline VectorizedN<data_t, 1> apply(
      const data_t* ptr,
      const VecMask<mask_t, 1>& vec_mask) {
    auto all_ones = _mm512_set1_epi32(0xFFFFFFFF);
    auto mmask = _mm512_cmp_epi32_mask(
        vec_mask.template cast<int, 1>()[0], all_ones, _MM_CMPINT_EQ);
    auto zero = _mm256_set1_epi16(0);
    auto temp = _mm256_mask_loadu_epi16(zero, mmask, ptr);
    return Vectorized<data_t>(
        _mm512_inserti32x8(_mm512_castsi256_si512(temp), zero, 1));
  }
};

template <typename data_t, typename mask_t>
struct VecMaskLoad<
    data_t,
    1,
    mask_t,
    1,
    typename std::enable_if<
        std::is_same_v<data_t, int8_t> ||
        std::is_same_v<data_t, uint8_t>>::type> {
  static inline VectorizedN<data_t, 1> apply(
      const data_t* ptr,
      const VecMask<mask_t, 1>& vec_mask) {
    auto all_ones = _mm512_set1_epi32(0xFFFFFFFF);
    auto mmask = _mm512_cmp_epi32_mask(
        vec_mask.template cast<int, 1>()[0], all_ones, _MM_CMPINT_EQ);
    auto zero = _mm_set1_epi8(0);
    auto temp = _mm_mask_loadu_epi8(zero, mmask, ptr);
    return Vectorized<data_t>(
        _mm512_inserti64x2(_mm512_set1_epi32(0), temp, 0));
  }
};

template <typename mask_t>
struct VecMaskLoad<int64_t, 2, mask_t, 1> {
  static inline VectorizedN<int64_t, 2> apply(
      const int64_t* ptr,
      const VecMask<mask_t, 1>& vec_mask) {
    auto all_ones = _mm512_set1_epi32(0xFFFFFFFF);
    auto zero = _mm512_set1_epi64(0);
    auto mmask = _mm512_cmp_epi32_mask(
        vec_mask.template cast<int, 1>()[0], all_ones, _MM_CMPINT_EQ);
    at::vec::VectorizedN<int64_t, 2> result;
    result[0] = _mm512_mask_loadu_epi64(zero, (__mmask8)mmask, ptr);
    result[1] = _mm512_mask_loadu_epi64(zero, (__mmask8)(mmask >> 8), ptr + 8);
    return result;
  }
};

template <>
struct VecMaskCast<float, 1, int, 1> {
  static inline VecMask<float, 1> apply(const VecMask<int, 1>& vec_mask) {
    return VectorizedN<float, 1>(_mm512_castsi512_ps(vec_mask[0]));
  }
};

template <>
struct VecMaskCast<int, 1, float, 1> {
  static inline VecMask<int, 1> apply(const VecMask<float, 1>& vec_mask) {
    return VectorizedN<int, 1>(_mm512_castps_si512(vec_mask[0]));
  }
};

template <typename dst_t>
struct VecMaskCast<dst_t, 1, int64_t, 2> {
  static inline VecMask<dst_t, 1> apply(const VecMask<int64_t, 2>& vec_mask) {
    auto low = _mm512_cvtepi64_epi32(vec_mask[0]);
    auto high = _mm512_cvtepi64_epi32(vec_mask[1]);
    return VecMask<int, 1>(VectorizedN<int, 1>(_mm512_inserti32x8(
                               _mm512_castsi256_si512(low), high, 1)))
        .cast<dst_t, 1>();
  }
};

template <>
inline bool VecMask<int, 1>::all_zero() const {
  __mmask16 mask = _mm512_test_epi32_mask(mask_[0], mask_[0]);
  return mask == 0;
}

template <>
inline bool VecMask<float, 1>::all_zero() const {
  return cast<int, 1>().all_zero();
}

template <>
inline bool VecMask<int64_t, 2>::all_zero() const {
  return cast<int, 1>().all_zero();
}

template <>
inline bool VecMask<int, 1>::is_masked(int i) const {
  return _mm512_movepi32_mask(mask_[0]) & (1 << i);
}

template <>
inline bool VecMask<float, 1>::is_masked(int i) const {
  return cast<int, 1>().is_masked(i);
}

template <>
inline bool VecMask<int64_t, 2>::is_masked(int i) const {
  return cast<int, 1>().is_masked(i);
}

#endif

} // namespace CPU_CAPABILITY
} // namespace at::vec
