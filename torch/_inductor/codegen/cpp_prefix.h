#pragma once

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <omp.h>

#include <ATen/NumericUtils.h>
#include <ATen/core/PhiloxRNGEngine.h>
#include <ATen/native/BinaryOps.h>
#include <ATen/native/Math.h>

#include <c10/util/BFloat16.h>
#include <c10/util/BFloat16-math.h>
#include <c10/util/Half.h>

#if defined(CPU_CAPABILITY_AVX512) || defined(CPU_CAPABILITY_AVX2)
#define INDUCTOR_USE_VECTOR_TYPES() 1
#else
#define INDUCTOR_USE_VECTOR_TYPES() 0
#endif

#if INDUCTOR_USE_VECTOR_TYPES()
#include <ATen/cpu/vec/functional.h>
#include <ATen/cpu/vec/vec.h>
#endif

typedef at::Half half;
typedef at::BFloat16 bfloat16;

template <typename T>
struct Welford {
  T mean = T(0);
  T m2 = T(0);
  T weight = T(0);
};


template <typename T>
struct IsVecType: std::false_type {};

#if INDUCTOR_USE_VECTOR_TYPES()
template <typename T>
struct IsVecType<at::vec::Vectorized<T>>: std::true_type {};
#endif

template <typename T>
Welford<T> welford_combine(const Welford<T> &a, const Welford<T> &b) {
  if constexpr (!IsVecType<T>::value) {
    if (a.weight == 0) {
      return b;
    }
    if (b.weight == 0) {
      return a;
    }
  }
  auto delta = b.mean - a.mean;
  auto new_weight = a.weight + b.weight;
  auto wb_over_w = b.weight / new_weight;
  if constexpr (IsVecType<T>::value) {
    // Guard against division by zero
    wb_over_w = T::blendv(wb_over_w, T(0), new_weight == T(0));
  }
  auto result = Welford<T>{
    a.mean + delta * wb_over_w,
    a.m2 + b.m2 + delta * delta * a.weight * wb_over_w,
    new_weight
  };
  return result;
}

template <typename T>
Welford<T> welford_combine(const Welford<T> &acc, T data) {
  // Add a single data point
  auto delta = data - acc.mean;
  auto new_weight = acc.weight + T(1);
  auto new_mean = acc.mean + delta / new_weight;
  auto new_delta = data - new_mean;
  auto result = Welford<T>{
    new_mean,
    acc.m2 + delta * new_delta,
    new_weight
  };
  return result;
}


#if INDUCTOR_USE_VECTOR_TYPES()
template <typename scalar_t>
inline at::vec::Vectorized<scalar_t> vec_shuffle_down(at::vec::Vectorized<scalar_t> x, size_t n) {
  using Vec = at::vec::Vectorized<scalar_t>;
  alignas(alignof(Vec)) scalar_t array[Vec::size()];
  x.store(array);
  for (size_t i = 0; i + n < Vec::size(); i += 2 * n) {
    array[i] = array[i + n];
  }
  return Vec::loadu(array);
}

#ifdef CPU_CAPABILITY_AVX2
inline at::vec::Vectorized<float> vec_shuffle_down(at::vec::Vectorized<float> x, size_t n) {
  using vec_t = at::vec::Vectorized<float>;
#define SHUFFLE_MASK(z, y, x, w) ((z << 6) | (y << 4) | (x << 2) | w)
  switch (n) {
  case 1:
    return vec_t(_mm256_permute_ps(x, SHUFFLE_MASK(1, 1, 3, 3)));
  case 2:
    return vec_t(_mm256_permute_ps(x, SHUFFLE_MASK(2, 2, 2, 2)));
  case 4:
    return vec_t(_mm256_permute2f128_ps(x, x, SHUFFLE_MASK(1, 1, 1, 1)));
  }
  TORCH_CHECK(false, "Unhandled vec_shuffle_down value ", n);
}
#endif

template <typename scalar_t>
Welford<scalar_t> welford_vec_reduce_all(Welford<at::vec::Vectorized<scalar_t>> acc) {
  using Vec = at::vec::Vectorized<scalar_t>;
  for (size_t n = 1; n < Vec::size(); n *= 2) {
    auto shuffled = Welford<Vec>{
      vec_shuffle_down(acc.mean, n),
      vec_shuffle_down(acc.m2, n),
      vec_shuffle_down(acc.weight, n)
    };
    acc = welford_combine(acc, shuffled);
  }

  Welford<scalar_t> result;
  alignas(alignof(Vec)) scalar_t array[Vec::size()];
  acc.mean.store(array);
  result.mean = array[0];

  acc.m2.store(array);
  result.m2 = array[0];

  acc.weight.store(array);
  result.weight = array[0];

  return result;
}
#endif


template <typename T> inline T mod(T a, T b) { return a % b; }
template <> inline float mod(float a, float b) { return std::fmod(a, b); }
template <> inline double mod(double a, double b) { return std::fmod(a, b); }

template <typename scalar_t>
inline scalar_t max_propagate_nan(scalar_t a, scalar_t b) {
  if (at::_isnan(a)) {
    return a;
  }
  return a > b ? a : b;
}

template <typename scalar_t>
inline scalar_t min_propagate_nan(scalar_t a, scalar_t b) {
  if (at::_isnan(a)) {
    return a;
  }
  return a < b ? a : b;
}

constexpr float uint32_to_uniform_float(uint32_t value) {
  // maximum value such that `MAX_INT * scale < 1.0` (with float rounding)
  constexpr float scale = 4.6566127342e-10;
  return static_cast<float>(value & 0x7FFFFFFF) * scale;
}

float normalized_rand_cpu(uint32_t seed, uint32_t offset) {
  return uint32_to_uniform_float(at::Philox4_32(seed, 0, offset)());
}

float randn_cpu(uint32_t seed, uint32_t offset) {
  at::Philox4_32 engine(seed, 0, offset);
  return engine.randn(10);
}

uint64_t randint64_cpu(uint32_t seed, uint32_t offset, int64_t low, int64_t high) {
  auto gen = at::Philox4_32(seed, 0, offset);
  uint64_t r0 = gen();
  uint64_t r1 = gen();
  uint64_t result = r0 | (r1 << 32);
  return (result % static_cast<uint64_t>(high - low)) + low;
}

template <typename T> struct AsIntegerType { typedef T type; };
template <> struct AsIntegerType<float> { typedef uint32_t type; };
template <> struct AsIntegerType<double> { typedef uint64_t type; };
template <> struct AsIntegerType<bfloat16> { typedef uint16_t type; };

template <typename T>
typename std::enable_if<!std::is_reduced_floating_point<T>::value, T>::type
inline fetch_value(volatile T *addr) {
  return *addr;
}

template <typename T>
typename std::enable_if<std::is_reduced_floating_point<T>::value, T>::type
inline fetch_value(volatile T *addr) {
  return T(addr->x, T::from_bits());
}

template <typename T>
typename std::enable_if<!std::is_integral<T>::value>::type
atomic_add(volatile T *addr, T offset) {
  typedef typename AsIntegerType<T>::type alt_type;

  static_assert(sizeof(std::atomic<alt_type>) == sizeof(T),
                "std::atomic issue");

  alt_type expected;

  alt_type desired;

  std::atomic<alt_type> *atomic_addr = (std::atomic<alt_type> *)addr;
  do {
    T val = fetch_value(addr);
    reinterpret_cast<T *>(&expected)[0] = val;
    reinterpret_cast<T *>(&desired)[0] = val + offset;
  } while (!atomic_addr->compare_exchange_weak(expected, desired,
                                               std::memory_order_relaxed));
}

// Since C++20 float is supported by fetch_add, but the performance may not
// better than compare_exchange_weak, which can be checked by microbenchmark
// inductor_cpu_atomic.py
template <typename T>
typename std::enable_if<std::is_integral<T>::value>::type
atomic_add(volatile T *addr, T offset) {
  static_assert(sizeof(std::atomic<T>) == sizeof(T),
                "std::atomic issue");
  std::atomic<T> *atomic_addr = (std::atomic<T> *)addr;
  atomic_addr->fetch_add(offset, std::memory_order_relaxed);
}

// This function is used to convert bool or uint8 to float mask for
// vectorization. The caller needs to make sure the src represents TRUE/FALSE
// correctly.
template <typename T>
inline float flag_to_float_scalar(T src) {
  float ret;
  *(uint32_t*)(&ret) = src ? 0xFFFFFFFF : 0;
  return ret;
}

#if defined(CPU_CAPABILITY_AVX512) || defined(CPU_CAPABILITY_AVX2)


inline at::vec::Vectorized<float> masked_load(const float* src, at::vec::Vectorized<float> mask) {
  at::vec::Vectorized<float> zero_vec(0);
# if defined(CPU_CAPABILITY_AVX512)
    auto all_ones = _mm512_set1_epi32(0xFFFFFFFF);
    auto mmask = _mm512_cmp_epi32_mask(_mm512_castps_si512(mask), all_ones, _MM_CMPINT_EQ);
    return _mm512_mask_loadu_ps(zero_vec, mmask, src);
# else // AVX2
    auto all_ones = _mm256_set1_epi32(0xFFFFFFFF);
    auto mmask = _mm256_cmp_epi32_mask(_mm256_castps_si256(mask), all_ones, _MM_CMPINT_EQ);
    return _mm256_mask_loadu_ps(zero_vec, mmask, src);
# endif
}


inline at::vec::Vectorized<bfloat16> masked_load(const bfloat16* src, at::vec::Vectorized<float> mask) {
# if defined(CPU_CAPABILITY_AVX512)
  auto all_ones = _mm512_set1_epi32(0xFFFFFFFF);
  auto mmask = _mm512_cmp_epi32_mask(_mm512_castps_si512(mask), all_ones, _MM_CMPINT_EQ);
  auto zero = _mm256_set1_epi16(0);
  auto temp = _mm256_mask_loadu_epi16(zero, mmask, src);
  __at_align__ int16_t tmp_values1[16];
  _mm256_storeu_si256(reinterpret_cast<__m256i*>(tmp_values1), temp);
  __at_align__ int16_t result[32];
  std::memcpy(result, tmp_values1, 16 * sizeof(int16_t));
  return _mm512_loadu_si512(reinterpret_cast<const __m512i*>(result));
# else // AVX2
  auto all_ones = _mm256_set1_epi32(0xFFFFFFFF);
  auto mmask = _mm256_cmp_epi32_mask(_mm256_castps_si256(mask), all_ones, _MM_CMPINT_EQ);
  auto zero = _mm_set1_epi16(0);
  auto temp = _mm_mask_loadu_epi16(zero, mmask, src);
  __at_align__ int16_t tmp_values1[8];
  _mm_storeu_si128(reinterpret_cast<__m128i*>(tmp_values1), temp);
  __at_align__ int16_t result[16];
  std::memcpy(result, tmp_values1, 16 * sizeof(int16_t));
  return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(result));
# endif
}


template <typename T>
inline at::vec::Vectorized<float> flag_to_float_vec(const T* src) {
  __at_align__ float dst_tmp[at::vec::Vectorized<float>::size()];
  #pragma unroll
  for (int64_t i = 0; i < at::vec::Vectorized<float>::size(); i++) {
    dst_tmp[i] = flag_to_float_scalar(src[i]);
  }
  return at::vec::Vectorized<float>::loadu(dst_tmp);
}

template <typename scalar_t>
inline at::vec::Vectorized<float> cvt_lowp_fp_to_fp32(
    at::vec::Vectorized<scalar_t> src) {
  at::vec::Vectorized<float> res_vec1(0);
  at::vec::Vectorized<float> res_vec2(0);
  std::tie(res_vec1, res_vec2) = at::vec::convert_to_float<scalar_t>(src);
  return res_vec1;
}

template <typename scalar_t>
inline at::vec::Vectorized<scalar_t> cvt_fp32_to_lowp_fp(
    at::vec::Vectorized<float> src) {
  return at::vec::convert_from_float<scalar_t>(src, src);
}

inline at::vec::Vectorized<float> mask_convert_to_float(at::vec::Vectorized<float> src) {
  auto zeros = at::vec::Vectorized<float>(0);
  auto ones = at::vec::Vectorized<float>(1);
  return at::vec::Vectorized<float>::blendv(zeros, ones, src);
}

template <typename SRC>
inline at::vec::Vectorized<float> vec_convert_to_mask(at::vec::Vectorized<SRC> src) {
  assert(
      at::vec::Vectorized<float>::size() == at::vec::Vectorized<SRC>::size());
  at::vec::Vectorized<float> res_vec(0);
  __at_align__ float dst_tmp[at::vec::Vectorized<float>::size()];
  __at_align__ SRC src_tmp[at::vec::Vectorized<SRC>::size()];
  src.store(src_tmp);

#pragma unroll
  for (int i = 0; i < at::vec::Vectorized<float>::size(); i++) {
    *(uint32_t*)(dst_tmp + i) = src_tmp[i] ? 0xFFFFFFFF : 0;
  }

  return res_vec.loadu(dst_tmp);
}

template <typename SRC>
inline at::vec::Vectorized<float> to_float_mask(at::vec::Vectorized<SRC> src) {
  return vec_convert_to_mask(src);
}

template <>
inline at::vec::Vectorized<float> to_float_mask(at::vec::Vectorized<int> src) {
#if defined(CPU_CAPABILITY_AVX2)
  return at::vec::Vectorized<float>(_mm256_castsi256_ps(src));
#else
  return at::vec::Vectorized<float>(_mm512_castsi512_ps(src));
#endif
}

template <>
inline at::vec::Vectorized<float> to_float_mask(at::vec::Vectorized<float> src) {
  return src;
}
#endif
