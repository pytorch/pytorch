#pragma once

#include <ATen/cpu/vec/intrinsics.h>
#include <ATen/cpu/vec/vec_base.h>
#include <ATen/cpu/vec/vec_mask.h>

namespace at::vec {
inline namespace CPU_CAPABILITY {

#if defined(CPU_CAPABILITY_AVX2) && !defined(_MSC_VER)

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
    VectorizedN<mask_t, 2> tmp_vec;
    VectorizedN<T, dst_n> result;
    for (int i = 0; i < dst_n; i++) {
      tmp_vec[0] = vec_mask[2 * i];
      tmp_vec[1] = vec_mask[2 * i + 1];
      auto int64_mask = VecMask<mask_t, 2>(tmp_vec).template cast<int64_t, 2>();
      auto int_mask = int64_mask.template cast<int, 1>()[0];
      if constexpr (std::is_same_v<T, float>) {
        result[i] = Vectorized<T>(
            _mm256_maskload_ps(ptr + i * Vectorized<T>::size(), int_mask));
      } else {
        result[i] = Vectorized<T>(
            _mm256_maskload_epi32(ptr + i * Vectorized<T>::size(), int_mask));
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
    VectorizedN<T, dst_n> result;
#ifndef _MSC_VER
#pragma unroll
#endif
    for (int i = 0; i < dst_n; i++) {
      auto tmp_mask = VecMask<mask_t, 1>(vec_mask[i]);
      auto int_mask = tmp_mask.template cast<int, 1>()[0];
      if constexpr (std::is_same_v<T, float>) {
        result[i] = Vectorized<T>(
            _mm256_maskload_ps(ptr + i * Vectorized<T>::size(), int_mask));
      } else {
        result[i] = Vectorized<T>(
            _mm256_maskload_epi32(ptr + i * Vectorized<T>::size(), int_mask));
      }
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
    typename std::enable_if_t<
        std::is_same_v<T, int64_t> || std::is_same_v<T, double>>> {
  static inline VectorizedN<T, 2> apply(
      const T* ptr,
      const VecMask<mask_t, 1>& vec_mask) {
    auto int64_mask = vec_mask.template cast<int64_t, 2>();
    auto result = at::vec::VectorizedN<T, 2>();
    if constexpr (std::is_same_v<T, double>) {
      result[0] = _mm256_maskload_pd(ptr, int64_mask[0]);
      result[1] = _mm256_maskload_pd(
          ptr + at::vec::Vectorized<T>::size(), int64_mask[1]);
    } else {
      result[0] = _mm256_maskload_epi64(
          reinterpret_cast<const long long*>(ptr), int64_mask[0]);
      result[1] = _mm256_maskload_epi64(
          reinterpret_cast<const long long*>(
              ptr + at::vec::Vectorized<T>::size()),
          int64_mask[1]);
    }
    return result;
  }
};

// TODO: add specialization of VecMaskLoad for bfloat16/half and int8/uint8

template <int N>
struct VecMaskCast<float, N, int, N> {
  static inline VecMask<float, N> apply(const VecMask<int, N>& vec_mask) {
    VectorizedN<float, N> result;
#ifndef _MSC_VER
#pragma unroll
#endif
    for (int i = 0; i < N; ++i) {
      result[i] = _mm256_castsi256_ps(vec_mask[i]);
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
      result[i] = _mm256_castps_si256(vec_mask[i]);
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
      result[i] = _mm256_castpd_si256(vec_mask[i]);
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
      result[i] = _mm256_castsi256_pd(vec_mask[i]);
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

template <int N>
struct VecMaskCheck<int64_t, N> {
  static inline bool all_zero(const VectorizedN<int64_t, N>& vec_mask) {
    bool all_zero = true;
    for (int i = 0; i < N; ++i) {
      all_zero = all_zero && (_mm256_testz_si256(vec_mask[i], vec_mask[i]) > 0);
      if (!all_zero) {
        return all_zero;
      }
    }
    return all_zero;
  }

  static inline bool is_masked(const VectorizedN<int64_t, N>& vec_mask, int i) {
    for (int j = 0; j < N; ++j) {
      if (i < (j + 1) * 4) {
        return _mm256_movemask_pd(_mm256_castsi256_pd(vec_mask[j])) &
            (1 << (i - j * 4));
      }
    }
    return false;
  }

  static inline bool all_masked(const VectorizedN<int64_t, N>& vec_mask) {
    bool all_masked = true;
    for (int i = 0; i < N; ++i) {
      all_masked = all_masked &&
          (_mm256_movemask_pd(_mm256_castsi256_pd(vec_mask[i])) == 0x0f);
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
