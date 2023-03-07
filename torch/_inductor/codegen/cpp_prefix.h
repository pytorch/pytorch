#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <omp.h>

#include <ATen/core/PhiloxRNGEngine.h>
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

template <typename T> struct AsIntegerType { typedef T type; };
template <> struct AsIntegerType<float> { typedef uint32_t type; };
template <> struct AsIntegerType<double> { typedef uint64_t type; };

template <typename T> void atomic_add(volatile T *addr, T offset) {
  typedef typename AsIntegerType<T>::type alt_type;

  static_assert(sizeof(std::atomic<alt_type>) == sizeof(T),
                "std::atomic issue");

  alt_type expected;

  alt_type desired;

  std::atomic<alt_type> *atomic_addr = (std::atomic<alt_type> *)addr;
  do {
    T val = *addr;
    reinterpret_cast<T *>(&expected)[0] = val;
    reinterpret_cast<T *>(&desired)[0] = val + offset;
  } while (!atomic_addr->compare_exchange_weak(expected, desired,
                                               std::memory_order_relaxed));
}

// This function is used to convert bool or uint8 to float mask for
// vectorization. The caller needs to make sure the src represents TRUE/FALSE
// correctly.
template <typename T>
void flag_to_float(const T* src, float* dst, int64_t n) {
#pragma unroll
  for (int64_t i = 0; i < n; i++) {
    uint32_t* dst_u32 = (uint32_t*)dst;
    dst_u32[i] = *(src + i) ? 0xFFFFFFFF : 0;
  }
}

template <typename T, std::enable_if_t<std::is_same<T, bool>::value || std::is_same<T, uint8_t>::value, bool> = true>
void flag_to_float(T src, float* dst, int64_t n) {
#pragma unroll
  for (int64_t i = 0; i < n; i++) {
    uint32_t* dst_u32 = (uint32_t*)dst;
    dst_u32[i] = src ? 0xFFFFFFFF : 0;
  }
}

#if defined(CPU_CAPABILITY_AVX512) || defined(CPU_CAPABILITY_AVX2)
template <typename SRC>
inline at::vec::Vectorized<float> to_float_mask(at::vec::Vectorized<SRC>& src) {
  assert(
      at::vec::Vectorized<float>::size() == at::vec::Vectorized<SRC>::size());
  at::vec::Vectorized<float> res_vec(0);
#pragma unroll
  for (int i = 0; i < at::vec::Vectorized<float>::size(); i++) {
    res_vec[i] = src[i] ? 0xFFFFFFFF : 0;
  }
  return res_vec;
}

template <>
inline at::vec::Vectorized<float> to_float_mask(at::vec::Vectorized<int>& src) {
#if defined(CPU_CAPABILITY_AVX2)
  return at::vec::Vectorized<float>(_mm256_cvtepi32_ps(src));
#else
  return at::vec::Vectorized<float>(_mm512_cvtepi32_ps(src));
#endif
}
#endif
