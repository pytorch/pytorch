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
#endif
#elif defined(_MSC_VER)
/* Microsoft C/C++-compatible compiler */
#include <intrin.h>
#if _MSC_VER <= 1900
#define _mm256_extract_epi64(X, Y) (_mm_extract_epi64(_mm256_extractf128_si256(X, Y >> 1), Y % 2))
#define _mm256_extract_epi32(X, Y) (_mm_extract_epi32(_mm256_extractf128_si256(X, Y >> 2), Y % 4))
#define _mm256_extract_epi16(X, Y) (_mm_extract_epi16(_mm256_extractf128_si256(X, Y >> 3), Y % 8))
#define _mm256_extract_epi8(X, Y) (_mm_extract_epi8(_mm256_extractf128_si256(X, Y >> 4), Y % 16))
#endif
#elif defined(__GNUC__) && (defined(__ARM_NEON__) || defined(__aarch64__))
/* GCC-compatible compiler, targeting ARM with NEON */
#include <arm_neon.h>
#if defined(__ARM_FEATURE_SVE)
/* GCC-compatible compiler, targeting ARM with SVE */
#include <arm_sve.h>
#endif
#if defined (MISSING_ARM_VLD1)
#include <ATen/cpu/vec/vec256/missing_vld1_neon.h>
#elif defined (MISSING_ARM_VST1)
#include <ATen/cpu/vec/vec256/missing_vst1_neon.h>
#endif
#elif defined(__GNUC__) && defined(__IWMMXT__)
/* GCC-compatible compiler, targeting ARM with WMMX */
#include <mmintrin.h>
#elif defined(__s390x__)
// targets Z/architecture
// we will include vecintrin later
#elif defined(__GNUC__) && defined(__riscv_v_intrinsic)
#include <riscv_vector.h>
#elif (defined(__GNUC__) || defined(__xlC__)) &&                               \
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
