#pragma once

// We want to allow 128-bit wide SIMD if either NEON is available (as
// detected by GEMMLOWP_NEON), or whether SSE4.2 and Clang is
// available (in which case we will use the neon_sse.h library to
// share source between the two implementations). We use SSE4.2 to
// ensure we can use the full neon2sse library, and we use Clang as
// GCC has issues correctly compiling some parts of the neon2sse
// library.

// Otherwise, the INT8_NEON_SIMD variable will be undefined.

#include "gemmlowp/fixedpoint/fixedpoint.h"
#include "gemmlowp/public/gemmlowp.h"

#ifdef GEMMLOWP_NEON
#define INT8_NEON_SIMD
#endif

#if defined(__SSE4_2__) && defined(__clang__)
#define INT8_NEON_SIMD

#include "neon2sse.h"
// Add GEMMLOWP SIMD type wrappers for the NEON2SSE SIMD types.

namespace gemmlowp {
template <>
struct FixedPointRawTypeTraits<int32x4_t> {
  typedef std::int32_t ScalarRawType;
  static const int kLanes = 4;
};

template <>
inline int32x4_t Dup<int32x4_t>(std::int32_t x) {
  return vdupq_n_s32(x);
}

template <>
inline int32x4_t BitAnd(int32x4_t a, int32x4_t b) {
  return vandq_s32(a, b);
}

template <>
inline int32x4_t Add(int32x4_t a, int32x4_t b) {
  return vaddq_s32(a, b);
}

template <>
inline int32x4_t ShiftRight(int32x4_t a, int offset) {
  return vshlq_s32(a, vdupq_n_s32(-offset));
}

template <>
inline int32x4_t MaskIfLessThan(int32x4_t a, int32x4_t b) {
  return vreinterpretq_s32_u32(vcltq_s32(a, b));
}

template <>
inline int32x4_t MaskIfGreaterThan(int32x4_t a, int32x4_t b) {
  return vreinterpretq_s32_u32(vcgtq_s32(a, b));
}

template <>
inline int32x4_t BitNot(int32x4_t a) {
  return veorq_s32(a, vdupq_n_s32(-1));
}

template <>
inline int32x4_t ShiftLeft(int32x4_t a, int offset) {
  return vshlq_s32(a, vdupq_n_s32(offset));
}

template <>
inline int32x4_t MaskIfZero(int32x4_t a) {
  return MaskIfEqual(a, vdupq_n_s32(0));
}

template <>
inline int32x4_t MaskIfNonZero(int32x4_t a) {
  return vreinterpretq_s32_u32(vtstq_s32(a, a));
}

template <>
inline int32x4_t SaturatingRoundingDoublingHighMul(int32x4_t a, int32x4_t b) {
  return vqrdmulhq_s32(a, b);
}

template <>
inline int32x4_t RoundingHalfSum(int32x4_t a, int32x4_t b) {
  return vrhaddq_s32(a, b);
}

} // namespace gemmlowp
#endif
