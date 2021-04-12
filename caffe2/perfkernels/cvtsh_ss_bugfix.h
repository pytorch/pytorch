#pragma once

// Apple clang was fixed in 8.1
#if defined(__apple_build_version__) && ((__clang_major__ < 8) || ((__clang_major__ == 8) && (__clang_minor__ < 1)))
#define CAFFE2_INTERNAL_APPLE_NEED_FIX 1
#endif

// Regular clang was fixed in 3.9
#if defined(__clang__) && (__clang_major__ < 4) && (__clang_minor__ < 9)
#define CAFFE2_INTERNAL_CLANG_NEED_FIX 1
#endif

#if defined(CAFFE2_INTERNAL_APPLE_NEED_FIX) || defined(CAFFE2_INTERNAL_CLANG_NEED_FIX)

#include <c10/util/Half.h>
#include <emmintrin.h>

// This version of clang has a bug that _cvtsh_ss is not defined, see
// https://reviews.llvm.org/D16177
static __inline float
    __attribute__((__always_inline__, __nodebug__, __target__("f16c")))
_cvtsh_ss(unsigned short a)
{
  __v8hi v = {(short)a, 0, 0, 0, 0, 0, 0, 0};
  __v4sf r = __builtin_ia32_vcvtph2ps(v);
  return r[0];
}

static __inline unsigned short
    __attribute__((__always_inline__, __nodebug__, __target__("f16c")))
_cvtss_sh(float a, int imm8) {
  unsigned short ret;
  *reinterpret_cast<at::Half*>(&ret) = a;
  return ret;
}

#endif // __APPLE_NEED_FIX || __CLANG_NEED_FIX

#undef __APPLE_NEED_FIX
#undef __CLANG_NEED_FIX

#if defined(_MSC_VER) && !defined(__clang__)

#include <c10/util/Half.h>
#include <cstdint>

// It seems that microsoft msvc does not have a _cvtsh_ss implementation so
// we will add a dummy version to it.

static inline float _cvtsh_ss(unsigned short x) {
  union {
    std::uint32_t intval;
    float floatval;
  } t1;
  std::uint32_t t2, t3;
  t1.intval = x & 0x7fff; // Non-sign bits
  t2 = x & 0x8000; // Sign bit
  t3 = x & 0x7c00; // Exponent
  t1.intval <<= 13; // Align mantissa on MSB
  t2 <<= 16; // Shift sign bit into position
  t1.intval += 0x38000000; // Adjust bias
  t1.intval = (t3 == 0 ? 0 : t1.intval); // Denormals-as-zero
  t1.intval |= t2; // Re-insert sign bit
  return t1.floatval;
}

static inline unsigned short _cvtss_sh(float x, int imm8) {
  unsigned short ret;
  *reinterpret_cast<at::Half*>(&ret) = x;
  return ret;
}

#endif // _MSC_VER
