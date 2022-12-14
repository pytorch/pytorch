#pragma once

// Defines the bloat16 type (brain floating-point). This representation uses
// 1 bit for the sign, 8 bits for the exponent and 7 bits for the mantissa.

#include <c10/macros/Macros.h>
#include <cmath>
#include <cstring>

#if defined(CUDA_VERSION) && CUDA_VERSION >= 11000
#include <cuda_bf16.h>
#endif

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

namespace c10 {

namespace detail {
inline C10_HOST_DEVICE float f32_from_bits(uint16_t src) {
  float res = 0;
  uint32_t tmp = src;
  tmp <<= 16;

#if defined(USE_ROCM)
  float* tempRes;

  // We should be using memcpy in order to respect the strict aliasing rule
  // but it fails in the HIP environment.
  tempRes = reinterpret_cast<float*>(&tmp);
  res = *tempRes;
#else
  std::memcpy(&res, &tmp, sizeof(tmp));
#endif

  return res;
}

inline C10_HOST_DEVICE uint16_t bits_from_f32(float src) {
  uint32_t res = 0;

#if defined(USE_ROCM)
  // We should be using memcpy in order to respect the strict aliasing rule
  // but it fails in the HIP environment.
  uint32_t* tempRes = reinterpret_cast<uint32_t*>(&src);
  res = *tempRes;
#else
  std::memcpy(&res, &src, sizeof(res));
#endif

  return res >> 16;
}

inline C10_HOST_DEVICE uint16_t round_to_nearest_even(float src) {
#if defined(USE_ROCM)
  if (src != src) {
#elif defined(_MSC_VER)
  if (isnan(src)) {
#else
  if (std::isnan(src)) {
#endif
    return UINT16_C(0x7FC0);
  } else {
    union {
      uint32_t U32;
      float F32;
    };

    F32 = src;
    uint32_t rounding_bias = ((U32 >> 16) & 1) + UINT32_C(0x7FFF);
    return static_cast<uint16_t>((U32 + rounding_bias) >> 16);
  }
}

inline C10_HOST_DEVICE void array_cvt_from_f32(const float *src, uint16_t *dst, size_t len) {
#if defined(__ARM_NEON) && defined(__ARM_BF16)
  float32x4_t s1;
  bfloat16x4_t d1;
  size_t rest = len % 4;
  size_t i = 0;
  for (i = 0; i < len - rest; i += 4) {  // 4 elems
    s1 = vld1q_f32(src+i);
    d1 = vcvt_bf16_f32(s1);
    vst1_u16(dst+i, vreinterpret_u16_bf16(d1));
  }
  while (i < len) {  // tail case
    bfloat16_t tmp = vcvth_bf16_f32(src[i]);
    std::memcpy(dst+i, &tmp, sizeof(tmp));
    i++;
  }
#else
  for (size_t i = 0; i < len; i++) {
    dst[i] = c10::detail::round_to_nearest_even(src[i]);
  }
#endif
}

} // namespace detail

struct alignas(2) BFloat16 {
  uint16_t x;

  // HIP wants __host__ __device__ tag, CUDA does not
#if defined(USE_ROCM)
  C10_HOST_DEVICE BFloat16() = default;
#else
  BFloat16() = default;
#endif

  struct from_bits_t {};
  static constexpr C10_HOST_DEVICE from_bits_t from_bits() {
    return from_bits_t();
  }

  constexpr C10_HOST_DEVICE BFloat16(unsigned short bits, from_bits_t)
      : x(bits){};
  inline C10_HOST_DEVICE BFloat16(float value);
  inline C10_HOST_DEVICE operator float() const;

#if defined(CUDA_VERSION) && CUDA_VERSION >= 11000
  inline C10_HOST_DEVICE BFloat16(const __nv_bfloat16& value);
  explicit inline C10_HOST_DEVICE operator __nv_bfloat16() const;
#endif
};

} // namespace c10

#include <c10/util/BFloat16-inl.h> // IWYU pragma: keep
