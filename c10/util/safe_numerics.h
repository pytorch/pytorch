#pragma once
#include <c10/macros/Macros.h>
#include <c10/util/ArrayRef.h>

#include <iterator>
#include <numeric>
#include <type_traits>

// GCC has __builtin_mul_overflow from before it supported __has_builtin
#ifdef __GNUC__
#define C10_HAS_BUILTIN_OVERFLOW() (1)
#elif defined(__has_builtin)
#define C10_HAS_BUILTIN_OVERFLOW()          \
  (__has_builtin(__builtin_mul_overflow) && \
   __has_builtin(__builtin_add_overflow))
#else
#define C10_HAS_BUILTIN_OVERFLOW() (0)
#endif

#if defined(_MSC_VER)
#include <c10/util/llvmMathExtras.h>
#include <intrin.h>
#endif

namespace c10 {

C10_ALWAYS_INLINE bool add_overflows(uint64_t a, uint64_t b, uint64_t* out) {
#if C10_HAS_BUILTIN_OVERFLOW()
  return __builtin_add_overflow(a, b, out);
#elif defined(_MSC_VER)
  unsigned long long tmp;
  auto carry = _addcarry_u64(0, a, b, &tmp);
  *out = tmp;
  return carry;
#else
  auto result = a + b;
  *out = result;
  return result < (a | b);
#endif
}

C10_ALWAYS_INLINE bool mul_overflows(uint64_t a, uint64_t b, uint64_t* out) {
#if C10_HAS_BUILTIN_OVERFLOW()
  return __builtin_mul_overflow(a, b, out);
#elif defined(_MSC_VER)
  *out = a * b;
  // This test isnt exact, but avoids doing integer division
  return (
      (c10::llvm::countLeadingZeros(a) + c10::llvm::countLeadingZeros(b)) < 64);
#else
  auto result = a * b;
  *out = result;
  return (a != 0) && (result / a != b);
#endif
}

} // namespace c10
