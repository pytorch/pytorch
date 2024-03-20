#pragma once

#include <ATen/cpu/vec/intrinsics.h>
#include <ATen/cpu/vec/vec_base.h>
#include <ATen/cpu/vec/vec_mask.h>

namespace at::vec {
inline namespace CPU_CAPABILITY {

#if defined(CPU_CAPABILITY_AVX2)

template <typename mask_t>
struct VecMaskLoad<float, 1, mask_t, 1> {
  static inline VectorizedN<float, 1> apply(
      const float* ptr,
      const VecMask<mask_t, 1>& vec_mask) {
    return Vectorized<float>(
        _mm256_maskload_ps(ptr, vec_mask.template cast<int, 1>()[0]));
  }
};

// TODO: add specialization of VecMaskLoad for bfloat16/half and int8/uint8

template <>
struct VecMaskCast<float, 1, int, 1> {
  static inline VecMask<float, 1> apply(const VecMask<int, 1>& vec_mask) {
    return VectorizedN<float, 1>(_mm256_castsi256_ps(vec_mask[0]));
  }
};

template <>
struct VecMaskCast<int, 1, float, 1> {
  static inline VecMask<int, 1> apply(const VecMask<float, 1>& vec_mask) {
    return VectorizedN<int, 1>(_mm256_castps_si256(vec_mask[0]));
  }
};

template <typename dst_t>
struct VecMaskCast<dst_t, 1, int64_t, 2> {
  static inline VecMask<dst_t, 1> apply(const VecMask<int64_t, 2>& vec_mask) {
    auto low = _mm256_shuffle_epi32(vec_mask[0], _MM_SHUFFLE(2, 0, 2, 0));
    auto high = _mm256_shuffle_epi32(vec_mask[1], _MM_SHUFFLE(2, 0, 2, 0));
    low = _mm256_permute4x64_epi64(low, _MM_SHUFFLE(3, 1, 2, 0));
    high = _mm256_permute4x64_epi64(high, _MM_SHUFFLE(3, 1, 2, 0));
    return VecMask<int, 1>(
        VectorizedN<int, 1>(_mm256_blend_epi32(low, high, 0xF0)));
  }
};

template <>
inline bool VecMask<int, 1>::all_zero() const {
  return _mm256_testz_si256(mask_[0], mask_[0]);
}

template <>
inline bool VecMask<float, 1>::all_zero() const {
  return _mm256_testz_ps(mask_[0], mask_[0]);
}

template <>
inline bool VecMask<int64_t, 2>::all_zero() const {
  return cast<int, 1>().all_zero();
}

template <>
inline bool VecMask<int, 1>::is_masked(int i) const {
  return _mm256_movemask_epi8(mask_[0]) & (1 << i);
}

template <>
inline bool VecMask<float, 1>::is_masked(int i) const {
  return _mm256_movemask_ps(mask_[0]) & (1 << i);
}

template <>
inline bool VecMask<int64_t, 2>::is_masked(int i) const {
  return cast<int, 1>().is_masked(i);
}

#endif

} // namespace CPU_CAPABILITY
} // namespace at::vec
