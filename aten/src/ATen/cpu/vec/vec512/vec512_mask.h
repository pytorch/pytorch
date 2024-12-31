#pragma once

#include <ATen/cpu/vec/intrinsics.h>
#include <ATen/cpu/vec/vec_base.h>
#include <ATen/cpu/vec/vec_mask.h>

namespace at::vec {
inline namespace CPU_CAPABILITY {

#if defined(CPU_CAPABILITY_AVX512) && !defined(_MSC_VER)

template <typename T, int dst_n, typename mask_t, int mask_n>
struct VecMaskLoad<
    T,
    dst_n,
    mask_t,
    mask_n,
    typename std::enable_if_t<
        (mask_n == dst_n * 2 && dst_n >= 1) &&
            (std::is_same_v<T, float> || std::is_same_v<T, int32_t>),
        void>> {
  static inline VectorizedN<T, dst_n> apply(
      const T* ptr,
      const VecMask<mask_t, mask_n>& vec_mask) {
    at::vec::Vectorized<T> zero_vec(0);
    auto all_ones = _mm512_set1_epi32(0xFFFFFFFF);
    VectorizedN<mask_t, 2> tmp_vec;
    VectorizedN<T, dst_n> result;
    for (int i = 0; i < dst_n; i++) {
      tmp_vec[0] = vec_mask[2 * i];
      tmp_vec[1] = vec_mask[2 * i + 1];
      auto int64_mask = VecMask<mask_t, 2>(tmp_vec).template cast<int64_t, 2>();
      auto int_mask = int64_mask.template cast<int, 1>()[0];
      auto mmask = _mm512_cmp_epi32_mask(int_mask, all_ones, _MM_CMPINT_EQ);
      if constexpr (std::is_same_v<T, float>) {
        result[i] = Vectorized<T>(_mm512_mask_loadu_ps(
            zero_vec, mmask, ptr + i * Vectorized<T>::size()));
      } else {
        result[i] = Vectorized<T>(_mm512_mask_loadu_epi32(
            zero_vec, mmask, ptr + i * Vectorized<T>::size()));
      }
    }
    return result;
  }
};

template <typename T, int dst_n, typename mask_t>
struct VecMaskLoad<
    T,
    dst_n,
    mask_t,
    dst_n,
    typename std::enable_if_t<
        std::is_same_v<T, float> || std::is_same_v<T, int32_t>,
        void>> {
  static inline VectorizedN<T, dst_n> apply(
      const T* ptr,
      const VecMask<mask_t, dst_n>& vec_mask) {
    at::vec::Vectorized<T> zero_vec(0);
    auto all_ones = _mm512_set1_epi32(0xFFFFFFFF);
    VectorizedN<T, dst_n> result;
#ifndef _MSC_VER
#pragma unroll
#endif
    for (int i = 0; i < dst_n; i++) {
      auto tmp_mask = VecMask<mask_t, 1>(vec_mask[i]);
      auto int_mask = tmp_mask.template cast<int, 1>()[0];
      auto mmask = _mm512_cmp_epi32_mask(int_mask, all_ones, _MM_CMPINT_EQ);
      if constexpr (std::is_same_v<T, float>) {
        result[i] = Vectorized<T>(_mm512_mask_loadu_ps(
            zero_vec, mmask, ptr + i * Vectorized<T>::size()));
      } else {
        result[i] = Vectorized<T>(_mm512_mask_loadu_epi32(
            zero_vec, mmask, ptr + i * Vectorized<T>::size()));
      }
    }
    return result;
  }
};

template <typename data_t, int dst_n, typename mask_t>
struct VecMaskLoad<
    data_t,
    dst_n,
    mask_t,
    dst_n,
    std::enable_if_t<
        std::is_same_v<data_t, BFloat16> ||
        std::is_same_v<data_t, Half>>> {
  static inline VectorizedN<data_t, dst_n> apply(
      const data_t* ptr,
      const VecMask<mask_t, dst_n>& vec_mask) {
    auto all_ones = _mm512_set1_epi32(0xFFFFFFFF);
    VectorizedN<data_t, dst_n> result;
#ifndef _MSC_VER
#pragma unroll
#endif
    for (int i = 0; i < dst_n; i++) {
      auto tmp_mask = VecMask<mask_t, 1>(vec_mask[i]);
      auto int_mask = tmp_mask.template cast<int, 2>();
      auto mmask0 = _mm512_cmp_epi32_mask(int_mask[0], all_ones, _MM_CMPINT_EQ);
      auto mmask1 = _mm512_cmp_epi32_mask(int_mask[1], all_ones, _MM_CMPINT_EQ);
      auto zero = _mm256_set1_epi16(0);
      auto temp0 = _mm256_mask_loadu_epi16(
          zero, mmask0, ptr + (2 * i) * Vectorized<int>::size());
      auto temp1 = _mm256_mask_loadu_epi16(
          zero, mmask1, ptr + (2 * i + 1) * Vectorized<int>::size());
      result[i] = Vectorized<data_t>(
          _mm512_inserti32x8(_mm512_castsi256_si512(temp0), temp1, 1));
    }
    return result;
  }
};

template <typename data_t, int dst_n, typename mask_t, int mask_n>
struct VecMaskLoad<
    data_t,
    dst_n,
    mask_t,
    mask_n,
    typename std::enable_if_t<
        (mask_n == 2 * dst_n && dst_n >= 1) &&
        (std::is_same_v<data_t, BFloat16> || std::is_same_v<data_t, Half>)>> {
  static inline VectorizedN<data_t, dst_n> apply(
      const data_t* ptr,
      const VecMask<mask_t, mask_n>& vec_mask) {
    auto all_ones = _mm512_set1_epi32(0xFFFFFFFF);
    VectorizedN<data_t, dst_n> result;
    VectorizedN<mask_t, 2> tmp_vec;
    for (int i = 0; i < dst_n; i++) {
      tmp_vec[0] = vec_mask[2 * i];
      tmp_vec[1] = vec_mask[2 * i + 1];
      auto int_mask = VecMask<mask_t, 2>(tmp_vec).template cast<int, 2>();
      auto mmask0 = _mm512_cmp_epi32_mask(int_mask[0], all_ones, _MM_CMPINT_EQ);
      auto mmask1 = _mm512_cmp_epi32_mask(int_mask[1], all_ones, _MM_CMPINT_EQ);
      auto zero = _mm256_set1_epi16(0);
      auto temp0 = _mm256_mask_loadu_epi16(
          zero, mmask0, ptr + (2 * i) * Vectorized<int>::size());
      auto temp1 = _mm256_mask_loadu_epi16(
          zero, mmask1, ptr + (2 * i + 1) * Vectorized<int>::size());
      result[i] = Vectorized<data_t>(
          _mm512_inserti32x8(_mm512_castsi256_si512(temp0), temp1, 1));
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
    std::enable_if_t<
        std::is_same_v<data_t, int8_t> ||
        std::is_same_v<data_t, uint8_t>>> {
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
    std::enable_if_t<
        std::is_same_v<data_t, int64_t> ||
        std::is_same_v<data_t, double>>> {
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
      result[1] =
          _mm512_mask_loadu_pd(zero_vec, (__mmask8)(mmask >> 8), ptr + 8);
    } else {
      result[0] = _mm512_mask_loadu_epi64(zero_vec, (__mmask8)mmask, ptr);
      result[1] =
          _mm512_mask_loadu_epi64(zero_vec, (__mmask8)(mmask >> 8), ptr + 8);
    }
    return result;
  }
};

template <int N>
struct VecMaskCast<float, N, int, N> {
  static inline VecMask<float, N> apply(const VecMask<int, N>& vec_mask) {
    VectorizedN<float, N> result;
#ifndef _MSC_VER
#pragma unroll
#endif
    for (int i = 0; i < N; ++i) {
      result[i] = _mm512_castsi512_ps(vec_mask[i]);
    }
    return result;
  }
};

template <int N>
struct VecMaskCast<int, N, float, N> {
  static inline VecMask<int, N> apply(const VecMask<float, N>& vec_mask) {
    VectorizedN<int, N> result;
#ifndef _MSC_VER
#pragma unroll
#endif
    for (int i = 0; i < N; ++i) {
      result[i] = _mm512_castps_si512(vec_mask[i]);
    }
    return result;
  }
};

template <int N>
struct VecMaskCast<int64_t, N, double, N> {
  static inline VecMask<int64_t, N> apply(const VecMask<double, N>& vec_mask) {
    VectorizedN<int64_t, N> result;
#ifndef _MSC_VER
#pragma unroll
#endif
    for (int i = 0; i < N; ++i) {
      result[i] = _mm512_castpd_si512(vec_mask[i]);
    }
    return result;
  }
};

template <int N>
struct VecMaskCast<double, N, int64_t, N> {
  static inline VecMask<double, N> apply(const VecMask<int64_t, N>& vec_mask) {
    VectorizedN<double, N> result;
#ifndef _MSC_VER
#pragma unroll
#endif
    for (int i = 0; i < N; ++i) {
      result[i] = _mm512_castsi512_pd(vec_mask[i]);
    }
    return result;
  }
};

template <int dst_n, typename mask_t, int mask_n>
struct VecMaskCast<
    int64_t,
    dst_n,
    mask_t,
    mask_n,
    typename std::enable_if_t<
        (dst_n == 2 * mask_n) &&
            (std::is_same_v<mask_t, float> || std::is_same_v<mask_t, int>),
        void>> {
  static inline VecMask<int64_t, dst_n> apply(
      const VecMask<mask_t, mask_n>& vec_mask) {
    VectorizedN<int64_t, dst_n> result;
    auto int_mask = vec_mask.template cast<int, mask_n>();
#ifndef _MSC_VER
#pragma unroll
#endif
    for (int i = 0; i < mask_n; ++i) {
      auto int64_vec =
          convert<int64_t, 2, int, 1>(VectorizedN<int, 1>(int_mask[i]));
      result[2 * i] = int64_vec[0];
      result[2 * i + 1] = int64_vec[1];
    }
    return VecMask<int64_t, dst_n>(result);
  }
};

template <typename dst_t, int dst_n, int mask_n>
struct VecMaskCast<
    dst_t,
    dst_n,
    int64_t,
    mask_n,
    typename std::enable_if_t<
        (mask_n == 2 * dst_n) &&
            (std::is_same_v<dst_t, float> || std::is_same_v<dst_t, int>),
        void>> {
  static inline VecMask<dst_t, dst_n> apply(
      const VecMask<int64_t, mask_n>& vec_mask) {
    VectorizedN<int, dst_n> result;
    VectorizedN<int64_t, 2> int64_vec;
    for (int i = 0; i < dst_n; ++i) {
      int64_vec[0] = vec_mask[2 * i];
      int64_vec[1] = vec_mask[2 * i + 1];
      result[i] = convert<int, 1, int64_t, 2>(int64_vec);
    }
    return VecMask<int, dst_n>(result).template cast<dst_t, dst_n>();
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

template <int N>
struct VecMaskCheck<int64_t, N> {
  static inline bool all_zero(const VectorizedN<int64_t, N>& vec_mask) {
    bool all_zero = true;
    for (int i = 0; i < N; ++i) {
      all_zero =
          all_zero && (_mm512_test_epi64_mask(vec_mask[i], vec_mask[i]) == 0);
      if (!all_zero) {
        return all_zero;
      }
    }
    return all_zero;
  }

  static inline bool is_masked(const VectorizedN<int64_t, N>& vec_mask, int i) {
    for (int j = 0; j < N; ++j) {
      if (i < (j + 1) * 8) {
        return _mm512_movepi64_mask(vec_mask[j]) & (1 << (i - j * 8));
      }
    }
    return false;
  }

  static inline bool all_masked(const VectorizedN<int64_t, N>& vec_mask) {
    bool all_masked = true;
    for (int i = 0; i < N; ++i) {
      all_masked = all_masked && (_mm512_movepi64_mask(vec_mask[i]) == 0xff);
      if (!all_masked) {
        return all_masked;
      }
    }
    return all_masked;
  }
};

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
