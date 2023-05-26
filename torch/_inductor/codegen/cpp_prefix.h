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

#if defined(CPU_CAPABILITY_AVX512) || defined(CPU_CAPABILITY_AVX2)
#include <ATen/cpu/vec/functional.h>
#include <ATen/cpu/vec/vec.h>
#endif
#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>

typedef at::Half half;
typedef at::BFloat16 bfloat16;

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
inline T fetch_value(volatile T *addr) {
  return *addr;
}

template <>
inline bfloat16 fetch_value<bfloat16>(volatile bfloat16 *addr) {
  return bfloat16(addr->x);
}

template <typename T> void atomic_add(volatile T *addr, T offset) {
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

template <typename T>
inline at::vec::Vectorized<float> flag_to_float_vec(const T* src) {
  __at_align__ float dst_tmp[at::vec::Vectorized<float>::size()];
  #pragma unroll
  for (int64_t i = 0; i < at::vec::Vectorized<float>::size(); i++) {
    dst_tmp[i] = flag_to_float_scalar(src[i]);
  }
  return at::vec::Vectorized<float>::loadu(dst_tmp);
}

inline at::vec::Vectorized<float> load_bf16_as_float(const bfloat16* bf16_buf) {
  at::vec::Vectorized<float> res_vec(0);
  at::vec::load_fp32_from_bf16(bf16_buf, res_vec);
  return res_vec;
}

inline void store_float_as_bf16(
    bfloat16* bf16_buf,
    at::vec::Vectorized<float> src_buf) {
  auto res = at::vec::convert_float_bfloat16(src_buf, src_buf);
  res.store(bf16_buf, at::vec::Vectorized<float>::size());
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
