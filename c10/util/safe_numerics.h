#pragma once
#include <c10/macros/Macros.h>
#include <c10/util/ArrayRef.h>

#include <iterator>
#include <numeric>
#include <type_traits>

// GCC has __builtin_mul_overflow from before it supported __has_builtin
#ifdef _MSC_VER
#define C10_HAS_BUILTIN_OVERFLOW() (0)
#include <c10/util/llvmMathExtras.h>
#include <intrin.h>
#else
#define C10_HAS_BUILTIN_OVERFLOW() (1)
#endif

namespace c10 {

C10_ALWAYS_INLINE bool add_overflows(uint64_t a, uint64_t b, uint64_t* out) {
#if C10_HAS_BUILTIN_OVERFLOW()
  return __builtin_add_overflow(a, b, out);
#else
  unsigned long long tmp;
#if defined(_M_IX86) || defined(_M_X64)
  auto carry = _addcarry_u64(0, a, b, &tmp);
#else
  tmp = a + b;
  unsigned long long vector = (a & b) ^ ((a ^ b) & ~tmp);
  auto carry = vector >> 63;
#endif
  *out = tmp;
  return carry;
#endif
}

C10_ALWAYS_INLINE bool mul_overflows(uint64_t a, uint64_t b, uint64_t* out) {
#if C10_HAS_BUILTIN_OVERFLOW()
  return __builtin_mul_overflow(a, b, out);
#else
  *out = a * b;
  // This test isnt exact, but avoids doing integer division
  return (
      (c10::llvm::countLeadingZeros(a) + c10::llvm::countLeadingZeros(b)) < 64);
#endif
}

C10_ALWAYS_INLINE bool mul_overflows(int64_t a, int64_t b, int64_t* out) {
#if C10_HAS_BUILTIN_OVERFLOW()
  return __builtin_mul_overflow(a, b, out);
#else
  volatile int64_t tmp = a * b;
  *out = tmp;
  if (a == 0 || b == 0) {
    return false;
  }
  return !(a == tmp / b);
#endif
}

template <typename It>
bool safe_multiplies_u64(It first, It last, uint64_t* out) {
#if C10_HAS_BUILTIN_OVERFLOW()
  uint64_t prod = 1;
  bool overflow = false;
  for (; first != last; ++first) {
    overflow |= c10::mul_overflows(prod, *first, &prod);
  }
  *out = prod;
  return overflow;
#else
  uint64_t prod = 1;
  uint64_t prod_log2 = 0;
  bool is_zero = false;
  for (; first != last; ++first) {
    auto x = static_cast<uint64_t>(*first);
    prod *= x;
    // log2(0) isn't valid, so need to track it specially
    is_zero |= (x == 0);
    prod_log2 += c10::llvm::Log2_64_Ceil(x);
  }
  *out = prod;
  // This test isnt exact, but avoids doing integer division
  return !is_zero && (prod_log2 >= 64);
#endif
}

template <typename Container>
bool safe_multiplies_u64(const Container& c, uint64_t* out) {
  return safe_multiplies_u64(c.begin(), c.end(), out);
}

} // namespace c10
