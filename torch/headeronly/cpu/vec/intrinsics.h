#pragma once
#if defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
/* GCC or clang-compatible compiler, targeting x86/x86-64 */
#include <x86intrin.h>
#elif defined(__clang__) && (defined(__ARM_NEON__) || defined(__aarch64__))
/* Clang-compatible compiler, targeting arm neon */
#include <arm_neon.h>
#if defined(__ARM_FEATURE_SVE)
/* CLANG-compatible compiler, targeting ARM with SVE */
#include <arm_sve.h>
#if __has_include(<arm_neon_sve_bridge.h>)
#include <arm_neon_sve_bridge.h>
#else
#define PYTORCH_NEON_SVE_BRIDGE_POLYFILL
#endif
#endif
#elif defined(_MSC_VER)
/* Microsoft C/C++-compatible compiler */
#include <intrin.h>
#if _MSC_VER <= 1900
#define _mm256_extract_epi64(X, Y) \
  (_mm_extract_epi64(_mm256_extractf128_si256(X, Y >> 1), Y % 2))
#define _mm256_extract_epi32(X, Y) \
  (_mm_extract_epi32(_mm256_extractf128_si256(X, Y >> 2), Y % 4))
#define _mm256_extract_epi16(X, Y) \
  (_mm_extract_epi16(_mm256_extractf128_si256(X, Y >> 3), Y % 8))
#define _mm256_extract_epi8(X, Y) \
  (_mm_extract_epi8(_mm256_extractf128_si256(X, Y >> 4), Y % 16))
#endif
#elif defined(__GNUC__) && (defined(__ARM_NEON__) || defined(__aarch64__))
/* GCC-compatible compiler, targeting ARM with NEON */
#include <arm_neon.h>
#if defined(__ARM_FEATURE_SVE)
/* GCC-compatible compiler, targeting ARM with SVE */
#include <arm_sve.h>
#if __has_include(<arm_neon_sve_bridge.h>)
#include <arm_neon_sve_bridge.h>
#else
#define PYTORCH_NEON_SVE_BRIDGE_POLYFILL
#endif
#endif
#elif defined(__GNUC__) && defined(__IWMMXT__)
/* GCC-compatible compiler, targeting ARM with WMMX */
#include <mmintrin.h>
#elif defined(__s390x__)
// targets Z/architecture
// we will include vecintrin later
#elif (defined(__GNUC__) || defined(__xlC__)) && \
    (defined(__VEC__) || defined(__ALTIVEC__))
/* XLC or GCC-compatible compiler, targeting PowerPC with VMX/VSX */
#include <altivec.h>
/* We need to undef those tokens defined by <altivec.h> to avoid conflicts
   with the C++ types. => Can still use __bool/__vector */
#undef bool
#undef vector
#undef pixel
#elif defined(__GNUC__) && defined(__SPE__)
/* GCC-compatible compiler, targeting PowerPC with SPE */
#include <spe.h>
#endif

#ifdef PYTORCH_NEON_SVE_BRIDGE_POLYFILL
// Polyfill for compilers without <arm_neon_sve_bridge.h> (e.g., GCC < 14).
// In VLS mode (-msve-vector-bits=N), SVE and NEON types share the same
// register file and layout, so these conversions compile to zero instructions.
#define PYTORCH_SVE_BRIDGE(neon_t, sve_t)                                    \
  static inline __attribute__((always_inline)) neon_t svget_neonq(sve_t v) { \
    neon_t r;                                                                \
    __builtin_memcpy(&r, &v, sizeof(r));                                     \
    return r;                                                                \
  }                                                                          \
  static inline __attribute__((always_inline)) sve_t svset_neonq(            \
      sve_t /*unused*/, neon_t v) {                                          \
    sve_t r;                                                                 \
    __builtin_memcpy(&r, &v, sizeof(v));                                     \
    return r;                                                                \
  }
PYTORCH_SVE_BRIDGE(float32x4_t, svfloat32_t)
PYTORCH_SVE_BRIDGE(int8x16_t, svint8_t)
PYTORCH_SVE_BRIDGE(uint8x16_t, svuint8_t)
PYTORCH_SVE_BRIDGE(int16x8_t, svint16_t)
PYTORCH_SVE_BRIDGE(int32x4_t, svint32_t)
PYTORCH_SVE_BRIDGE(int64x2_t, svint64_t)
PYTORCH_SVE_BRIDGE(float64x2_t, svfloat64_t)
#ifdef __ARM_FEATURE_BF16
PYTORCH_SVE_BRIDGE(bfloat16x8_t, svbfloat16_t)
#endif
#undef PYTORCH_SVE_BRIDGE
#undef PYTORCH_NEON_SVE_BRIDGE_POLYFILL
#endif
