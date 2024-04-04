#pragma once

#include <ATen/cpu/vec/intrinsics.h>
#include <ATen/cpu/vec/vec_base.h>
#include <ATen/cpu/vec/vec_mask.h>

namespace at::vec {
inline namespace CPU_CAPABILITY {

#if defined(CPU_CAPABILITY_AVX2) && !defined(_MSC_VER)

template <typename T, typename mask_t>
struct VecMaskLoad<
    T,
    1,
    mask_t,
    1,
    typename std::enable_if_t<
        std::is_same_v<T, float> || std::is_same_v<T, int32_t> ||
            std::is_same_v<T, uint32_t>,
        void>> {
  static inline VectorizedN<T, 1> apply(
      const T* ptr,
      const VecMask<mask_t, 1>& vec_mask) {
    auto int_mask = vec_mask.template cast<int, 1>()[0];
    if constexpr (std::is_same_v<T, float>) {
      return Vectorized<T>(_mm256_maskload_ps(ptr, int_mask));
    } else {
      return Vectorized<T>(_mm256_maskload_epi32(ptr, int_mask));
    }
  }
};

// TODO: add specialization of VecMaskLoad for bfloat16/half and int8/uint8

template <>
struct VecMaskCast<float, 1, int, 1> {
  static inline VecMask<float, 1> apply(const VecMask<int, 1>& vec_mask) {
    return Vectorized<float>(_mm256_castsi256_ps(vec_mask[0]));
  }
};

template <>
struct VecMaskCast<int, 1, float, 1> {
  static inline VecMask<int, 1> apply(const VecMask<float, 1>& vec_mask) {
    return Vectorized<int>(_mm256_castps_si256(vec_mask[0]));
  }
};

template <typename dst_t>
struct VecMaskCast<dst_t, 1, int64_t, 2> {
  static inline VecMask<dst_t, 1> apply(const VecMask<int64_t, 2>& vec_mask) {
    auto int_vec = convert<int, 1, int64_t, 2>(VectorizedN<int64_t, 2>(vec_mask));
    return VecMask<int, 1>(int_vec).cast<dst_t, 1>();
  }
};

template <>
inline bool VecMask<int, 1>::all_zero() const {
  return _mm256_testz_si256(mask_[0], mask_[0]);
}

template <>
inline bool VecMask<int, 1>::is_masked(int i) const {
  return _mm256_movemask_ps(_mm256_castsi256_ps(mask_[0])) & (1 << i);
}

template <>
inline bool VecMask<int, 1>::all_masked() const {
  int mask = _mm256_movemask_ps(_mm256_castsi256_ps(mask_[0]));
  return mask == 0xff;
}

#define VEC_MASK_METHOD_WITH_CAST_TO_INT(                   \
    T, N, return_type, method, args_def, args)              \
  template <>                                               \
  inline return_type VecMask<T, N>::method args_def const { \
    return cast<int, 1>().method args;                      \
  }

VEC_MASK_METHOD_WITH_CAST_TO_INT(float, 1, bool, all_zero, (), ())
VEC_MASK_METHOD_WITH_CAST_TO_INT(int64_t, 2, bool, all_zero, (), ())
VEC_MASK_METHOD_WITH_CAST_TO_INT(float, 1, bool, is_masked, (int i), (i))
VEC_MASK_METHOD_WITH_CAST_TO_INT(int64_t, 2, bool, is_masked, (int i), (i))
VEC_MASK_METHOD_WITH_CAST_TO_INT(float, 1, bool, all_masked, (), ())
VEC_MASK_METHOD_WITH_CAST_TO_INT(int64_t, 2, bool, all_masked, (), ())

#undef VEC_MASK_DEFINE_METHOD_WITH_CAST_TO_INT

#endif

} // namespace CPU_CAPABILITY
} // namespace at::vec
