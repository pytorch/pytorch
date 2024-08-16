#pragma once

#include <ATen/cpu/vec/intrinsics.h>
#include <ATen/cpu/vec/vec_base.h>
#include <ATen/cpu/vec/vec_mask.h>

namespace at::vec {
inline namespace CPU_CAPABILITY {

#if defined(CPU_CAPABILITY_AVX512) && !defined(_MSC_VER)

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
    at::vec::Vectorized<T> zero_vec(0);
    auto all_ones = _mm512_set1_epi32(0xFFFFFFFF);
    auto int_mask = vec_mask.template cast<int, 1>()[0];
    auto mmask = _mm512_cmp_epi32_mask(int_mask, all_ones, _MM_CMPINT_EQ);
    if constexpr (std::is_same_v<T, float>) {
      return Vectorized<T>(_mm512_mask_loadu_ps(zero_vec, mmask, ptr));
    } else {
      return Vectorized<T>(_mm512_mask_loadu_epi32(zero_vec, mmask, ptr));
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
    at::vec::Vectorized<T> zero_vec(0);
    auto all_ones = _mm512_set1_epi32(0xFFFFFFFF);
    auto int_mask = vec_mask.template cast<int, 2>();
    auto mmask0 = _mm512_cmp_epi32_mask(int_mask[0], all_ones, _MM_CMPINT_EQ);
    auto mmask1 = _mm512_cmp_epi32_mask(int_mask[1], all_ones, _MM_CMPINT_EQ);
    auto result = at::vec::VectorizedN<T, 2>();
    if constexpr (std::is_same_v<T, float>) {
      result[0] = _mm512_mask_loadu_ps(zero_vec, mmask0, ptr);
      result[1] = _mm512_mask_loadu_ps(zero_vec, mmask1, ptr + at::vec::Vectorized<T>::size());
    } else {
      result[0] = _mm512_mask_loadu_epi32(zero_vec, mmask0, ptr);
      result[1] = _mm512_mask_loadu_epi32(zero_vec, mmask1, ptr + at::vec::Vectorized<T>::size());
    }
    return result;
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
    auto int_mask = vec_mask.template cast<int, 1>()[0];
    auto mmask = _mm512_cmp_epi32_mask(int_mask, all_ones, _MM_CMPINT_EQ);
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
    2,
    typename std::enable_if<
        std::is_same_v<data_t, BFloat16> ||
        std::is_same_v<data_t, Half>>::type> {
  static inline VectorizedN<data_t, 1> apply(
      const data_t* ptr,
      const VecMask<mask_t, 2>& vec_mask) {
    auto all_ones = _mm512_set1_epi32(0xFFFFFFFF);
    auto int_mask = vec_mask.template cast<int, 2>();
    auto mmask0 = _mm512_cmp_epi32_mask(int_mask[0], all_ones, _MM_CMPINT_EQ);
    auto mmask1 = _mm512_cmp_epi32_mask(int_mask[1], all_ones, _MM_CMPINT_EQ);
    auto zero = _mm256_set1_epi16(0);
    auto temp0 = _mm256_mask_loadu_epi16(zero, mmask0, ptr);
    auto temp1 = _mm256_mask_loadu_epi16(zero, mmask1, ptr + Vectorized<mask_t>::size());
    return Vectorized<data_t>(
        _mm512_inserti32x8(_mm512_castsi256_si512(temp0), temp1, 1));
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
    auto int_mask = vec_mask.template cast<int, 1>()[0];
    auto mmask = _mm512_cmp_epi32_mask(int_mask, all_ones, _MM_CMPINT_EQ);
    auto zero = _mm_set1_epi8(0);
    auto temp = _mm_mask_loadu_epi8(zero, mmask, ptr);
    return Vectorized<data_t>(
        _mm512_inserti64x2(_mm512_set1_epi32(0), temp, 0));
  }
};

template <typename data_t, typename mask_t>
struct VecMaskLoad<
    data_t,
    2,
    mask_t,
    1,
    typename std::enable_if<
        std::is_same_v<data_t, int64_t> ||
        std::is_same_v<data_t, double>>::type> {
  static inline VectorizedN<data_t, 2> apply(
      const data_t* ptr,
      const VecMask<mask_t, 1>& vec_mask) {
    auto all_ones = _mm512_set1_epi32(0xFFFFFFFF);
    at::vec::Vectorized<data_t> zero_vec(0);
    auto int_mask = vec_mask.template cast<int, 1>()[0];
    auto mmask = _mm512_cmp_epi32_mask(int_mask, all_ones, _MM_CMPINT_EQ);
    at::vec::VectorizedN<data_t, 2> result;
    if constexpr (std::is_same_v<data_t, double>) {
      result[0] = _mm512_mask_loadu_pd(zero_vec, (__mmask8)mmask, ptr);
      result[1] = _mm512_mask_loadu_pd(zero_vec, (__mmask8)(mmask >> 8), ptr + 8);
    } else {
      result[0] = _mm512_mask_loadu_epi64(zero_vec, (__mmask8)mmask, ptr);
      result[1] = _mm512_mask_loadu_epi64(zero_vec, (__mmask8)(mmask >> 8), ptr + 8);
    }
    return result;
  }
};

template <>
struct VecMaskCast<float, 1, int, 1> {
  static inline VecMask<float, 1> apply(const VecMask<int, 1>& vec_mask) {
    return Vectorized<float>(_mm512_castsi512_ps(vec_mask[0]));
  }
};

template <>
struct VecMaskCast<int, 1, float, 1> {
  static inline VecMask<int, 1> apply(const VecMask<float, 1>& vec_mask) {
    return Vectorized<int>(_mm512_castps_si512(vec_mask[0]));
  }
};

template <>
struct VecMaskCast<float, 2, int, 2> {
  static inline VecMask<float, 2> apply(const VecMask<int, 2>& vec_mask) {
    return VectorizedN<float, 2>(_mm512_castsi512_ps(vec_mask[0]), _mm512_castsi512_ps(vec_mask[1]));
  }
};

template <>
struct VecMaskCast<int, 2, float, 2> {
  static inline VecMask<int, 2> apply(const VecMask<float, 2>& vec_mask) {
    return VectorizedN<int, 2>(_mm512_castps_si512(vec_mask[0]), _mm512_castps_si512(vec_mask[1]));
  }
};

template <>
struct VecMaskCast<int64_t, 2, double, 2> {
  static inline VecMask<int64_t, 2> apply(const VecMask<double, 2>& vec_mask) {
    VectorizedN<int64_t, 2> result;
    result[0] = _mm512_castpd_si512(vec_mask[0]);
    result[1] = _mm512_castpd_si512(vec_mask[1]);
    return result;
  }
};

template <>
struct VecMaskCast<double, 2, int64_t, 2> {
  static inline VecMask<double, 2> apply(const VecMask<int64_t, 2>& vec_mask) {
    VectorizedN<double, 2> result;
    result[0] = _mm512_castsi512_pd(vec_mask[0]);
    result[1] = _mm512_castsi512_pd(vec_mask[1]);
    return result;
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

template <typename dst_t>
struct VecMaskCast<dst_t, 1, int64_t, 2> {
  static inline VecMask<dst_t, 1> apply(const VecMask<int64_t, 2>& vec_mask) {
    auto int_vec = convert<int, 1, int64_t, 2>(VectorizedN<int64_t, 2>(vec_mask));
    return VecMask<int, 1>(int_vec).cast<dst_t, 1>();
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
  __mmask16 mask = _mm512_test_epi32_mask(mask_[0], mask_[0]);
  return mask == 0;
}

template <>
inline bool VecMask<int, 1>::is_masked(int i) const {
  return _mm512_movepi32_mask(mask_[0]) & (1 << i);
}

template <>
inline bool VecMask<int, 1>::all_masked() const {
  __mmask16 mask = _mm512_movepi32_mask(mask_[0]);
  return mask == 0xffff;
}

template <>
inline bool VecMask<int64_t, 4>::all_zero() const {
  __mmask8 mask0 = _mm512_test_epi64_mask(mask_[0], mask_[0]);
  __mmask8 mask1 = _mm512_test_epi64_mask(mask_[1], mask_[1]);
  __mmask8 mask2 = _mm512_test_epi64_mask(mask_[2], mask_[2]);
  __mmask8 mask3 = _mm512_test_epi64_mask(mask_[3], mask_[3]);
  return mask0 == 0 && mask1 == 0 && mask2 == 0 && mask3 == 0;
}

template <>
inline bool VecMask<int64_t, 4>::is_masked(int i) const {
  if (i < 8) {
    return _mm512_movepi64_mask(mask_[0]) & (1 << i);
  } else if (i < 16) {
    return _mm512_movepi64_mask(mask_[1]) & (1 << (i - 8));
  } else if (i < 24) {
    return _mm512_movepi64_mask(mask_[2]) & (1 << (i - 16));
  } else {
    return _mm512_movepi64_mask(mask_[3]) & (1 << (i - 24));
  }
}

template <>
inline bool VecMask<int64_t, 4>::all_masked() const {
  __mmask8 mask0 = _mm512_movepi64_mask(mask_[0]);
  __mmask8 mask1 = _mm512_movepi64_mask(mask_[1]);
  __mmask8 mask2 = _mm512_movepi64_mask(mask_[2]);
  __mmask8 mask3 = _mm512_movepi64_mask(mask_[3]);
  return mask0 == 0xff && mask1 == 0xff && mask2 == 0xff && mask3 == 0xff;
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
