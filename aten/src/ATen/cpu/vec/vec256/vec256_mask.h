#pragma once

#include <ATen/cpu/vec/intrinsics.h>
#include <ATen/cpu/vec/vec_base.h>
#include <ATen/cpu/vec/vec_mask.h>

namespace at::vec {
inline namespace CPU_CAPABILITY {

#if defined(CPU_CAPABILITY_AVX2) && !defined(_MSC_VER)

template <typename T, typename mask_t, int mask_n>
struct VecMaskLoad<
    T,
    1,
    mask_t,
    mask_n,
    typename std::enable_if_t<
        (mask_n == 1 || mask_n == 2) &&
        (std::is_same_v<T, float> || std::is_same_v<T, int32_t> ||
            std::is_same_v<T, uint32_t>),
        void>> {
  static inline VectorizedN<T, 1> apply(
      const T* ptr,
      const VecMask<mask_t, mask_n>& vec_mask) {
    auto int_mask = vec_mask.template cast<int, 1>()[0];
    if constexpr (std::is_same_v<T, float>) {
      return Vectorized<T>(_mm256_maskload_ps(ptr, int_mask));
    } else {
      return Vectorized<T>(_mm256_maskload_epi32(ptr, int_mask));
    }
  }
};

template <typename T, typename mask_t, int mask_n>
struct VecMaskLoad<
    T,
    2,
    mask_t,
    mask_n,
    typename std::enable_if_t<
        (mask_n == 2 || mask_n == 4) &&
        (std::is_same_v<T, float> || std::is_same_v<T, int32_t> ||
            std::is_same_v<T, uint32_t>),
        void>> {
  static inline VectorizedN<T, 2> apply(
      const T* ptr,
      const VecMask<mask_t, mask_n>& vec_mask) {
    auto int_mask = vec_mask.template cast<int, 2>();
    auto result = at::vec::VectorizedN<T, 2>();
    if constexpr (std::is_same_v<T, float>) {
      result[0] = _mm256_maskload_ps(ptr, int_mask[0]);
      result[1] = _mm256_maskload_ps(ptr + at::vec::Vectorized<T>::size(), int_mask[1]);
    } else {
      result[0] = _mm256_maskload_epi32(ptr, int_mask[0]);
      result[1] = _mm256_maskload_epi32(ptr + at::vec::Vectorized<T>::size(), int_mask[1]);
    }
    return result;
  }
};

template <typename T, typename mask_t>
struct VecMaskLoad<
    T,
    2,
    mask_t,
    1,
    typename std::enable_if<
        (std::is_same_v<T, int64_t> && sizeof(int64_t) == sizeof(long long)) ||
        std::is_same_v<T, double>>::type> {
  static inline VectorizedN<T, 2> apply(
      const T* ptr,
      const VecMask<mask_t, 1>& vec_mask) {
    auto int64_mask = vec_mask.template cast<int64_t, 2>();
    auto result = at::vec::VectorizedN<T, 2>();
    if constexpr (std::is_same_v<T, double>) {
      result[0] = _mm256_maskload_pd(ptr, int64_mask[0]);
      result[1] = _mm256_maskload_pd(ptr + at::vec::Vectorized<T>::size(), int64_mask[1]);
    } else {
      result[0] = _mm256_maskload_epi64(ptr, int64_mask[0]);
      result[1] = _mm256_maskload_epi64(ptr + at::vec::Vectorized<T>::size(), int64_mask[1]);
    }
    return result;
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
template <>
struct VecMaskCast<float, 2, int, 2> {
  static inline VecMask<float, 2> apply(const VecMask<int, 2>& vec_mask) {
    return VectorizedN<float, 2>(_mm256_castsi256_ps(vec_mask[0]), _mm256_castsi256_ps(vec_mask[1]));
  }
};
template <>
struct VecMaskCast<int, 2, float, 2> {
  static inline VecMask<int, 2> apply(const VecMask<float, 2>& vec_mask) {
    return VectorizedN<int, 2>(_mm256_castps_si256(vec_mask[0]), _mm256_castps_si256(vec_mask[1]));
  }
};

template <>
struct VecMaskCast<int64_t, 2, double, 2> {
  static inline VecMask<int64_t, 2> apply(const VecMask<double, 2>& vec_mask) {
    VectorizedN<int64_t, 2> result;
    result[0] = _mm256_castpd_si256(vec_mask[0]);
    result[1] = _mm256_castpd_si256(vec_mask[1]);
    return result;
  }
};

template <>
struct VecMaskCast<double, 2, int64_t, 2> {
  static inline VecMask<double, 2> apply(const VecMask<int64_t, 2>& vec_mask) {
    VectorizedN<double, 2> result;
    result[0] = _mm256_castsi256_pd(vec_mask[0]);
    result[1] = _mm256_castsi256_pd(vec_mask[1]);
    return result;
  }
};
template <typename dst_t>
struct VecMaskCast<dst_t, 1, int64_t, 2> {
  static inline VecMask<dst_t, 1> apply(const VecMask<int64_t, 2>& vec_mask) {
    auto int_vec = convert<int, 1, int64_t, 2>(VectorizedN<int64_t, 2>(vec_mask));
    return VecMask<int, 1>(int_vec).cast<dst_t, 1>();
  }
};
template <typename mask_t>
struct VecMaskCast<int64_t, 2, mask_t, 1> {
  static inline VecMask<int64_t, 2> apply(const VecMask<mask_t, 1>& vec_mask) {
    auto int_mask = vec_mask.template cast<int, 1>();
    auto int64_vec = convert<int64_t, 2, int, 1>(VectorizedN<int, 1>(int_mask[0]));
    return int64_vec;
  }
};
template <>
struct VecMaskCast<double, 2, float, 1> {
  static inline VecMask<double, 2> apply(const VecMask<float, 1>& vec_mask) {
    auto int64_mask = VecMaskCast<int64_t, 2, float, 1>::apply(vec_mask);
    return VecMaskCast<double, 2, int64_t, 2>::apply(int64_mask);
  }
};
template <>
struct VecMaskCast<float, 1, double, 2> {
  static inline VecMask<float, 1> apply(const VecMask<double, 2>& vec_mask) {
    auto int64_mask = VecMaskCast<int64_t, 2, double, 2>::apply(vec_mask);
    return VecMaskCast<float, 1, int64_t, 2>::apply(int64_mask);
  }
};

template <typename mask_t>
struct VecMaskCast<int64_t, 4, mask_t, 2> {
  static inline VecMask<int64_t, 4> apply(const VecMask<mask_t, 2>& vec_mask) {
    auto result = at::vec::VectorizedN<int64_t, 4>();
    auto int_mask = vec_mask.template cast<int, 2>();
    auto int64_vec = convert<int64_t, 2, int, 1>(VectorizedN<int, 1>(int_mask[0]));
    result[0] = int64_vec[0];
    result[1] = int64_vec[1];
    int64_vec = convert<int64_t, 2, int, 1>(VectorizedN<int, 1>(int_mask[1]));
    result[2] = int64_vec[0];
    result[3] = int64_vec[1];
    return VecMask<int64_t, 4>(result);
  }
};

template <typename dst_t>
struct VecMaskCast<dst_t, 2, int64_t, 4> {
  static inline VecMask<dst_t, 2> apply(const VecMask<int64_t, 4>& vec_mask) {
    auto result = VecMask<int, 2>();
    auto int64_vec = at::vec::VectorizedN<int64_t, 2>();
    int64_vec[0] = vec_mask[0];
    int64_vec[1] = vec_mask[1];
    result[0] = convert<int, 1, int64_t, 2>(int64_vec);
    int64_vec[0] = vec_mask[2];
    int64_vec[1] = vec_mask[3];
    result[1] = convert<int, 1, int64_t, 2>(int64_vec);
    return VecMask<int, 2>(result).cast<dst_t, 2>();
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

template <>
inline bool VecMask<int64_t, 4>::all_zero() const {
  return _mm256_testz_si256(mask_[0], mask_[0]) &&
         _mm256_testz_si256(mask_[1], mask_[1]) &&
         _mm256_testz_si256(mask_[2], mask_[2]) &&
         _mm256_testz_si256(mask_[3], mask_[3]);
}

template <>
inline bool VecMask<int64_t, 4>::is_masked(int i) const {
  if (i < 4) {
    return _mm256_movemask_pd(_mm256_castsi256_pd(mask_[0])) & (1 << i);
  } else if (i < 8) {
    return _mm256_movemask_pd(_mm256_castsi256_pd(mask_[0])) & (1 << (i - 4));
  } else if (i < 12) {
    return _mm256_movemask_pd(_mm256_castsi256_pd(mask_[0])) & (1 << (i - 8));
  } else {
    return _mm256_movemask_pd(_mm256_castsi256_pd(mask_[0])) & (1 << (i - 12));
  }
}

template <>
inline bool VecMask<int64_t, 4>::all_masked() const {
  int mask0 = _mm256_movemask_pd(_mm256_castsi256_pd(mask_[0]));
  int mask1 = _mm256_movemask_pd(_mm256_castsi256_pd(mask_[1]));
  int mask2 = _mm256_movemask_pd(_mm256_castsi256_pd(mask_[2]));
  int mask3 = _mm256_movemask_pd(_mm256_castsi256_pd(mask_[3]));
  return mask0 == 0x0f && mask1 == 0x0f && mask2 == 0x0f && mask3 == 0x0f;
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
