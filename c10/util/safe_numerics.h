#pragma once
#include <c10/macros/Macros.h>

#include <cstddef>
#include <cstdint>

// GCC has __builtin_mul_overflow from before it supported __has_builtin
#ifdef _MSC_VER
#define C10_HAS_BUILTIN_OVERFLOW() (0)
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

C10_ALWAYS_INLINE bool mul_overflows(int64_t a, int64_t b, int64_t* out) {
#if C10_HAS_BUILTIN_OVERFLOW()
  return __builtin_mul_overflow(a, b, out);
#else
  int64_t high{0};
  *out = _mul128(a, b, &high);
  // Idea: Check if int128 represented as (high, low) can
  // be stored in an int64. This is possible only when the
  // high bits are all sign extension bits.
  //
  // Implementation: All of the bits in high should be the
  // same as the top bit (sign bit) of low. (low>>63) does
  // a sign extension and produces a 64 bit number with all
  // bits equal to the sign bit. XOR'ing this with high lets
  // us compare if they are same or different.
  return (high ^ (*out >> 63));
#endif
}

C10_ALWAYS_INLINE bool mul_overflows(uint64_t a, uint64_t b, uint64_t* out) {
#if C10_HAS_BUILTIN_OVERFLOW()
  return __builtin_mul_overflow(a, b, out);
#else
  uint64_t high{0};
  *out = _umul128(a, b, &high);
  // If all the high bits are 0 then the int128 can be safely
  // converted to an int64 just by keeping the low bits
  return static_cast<bool>(high);
#endif
}

template <typename It>
bool safe_multiplies_u64(It first, It last, uint64_t* out) {
  uint64_t prod = 1;
  bool overflow = false;
  for (; first != last; ++first) {
    overflow |= c10::mul_overflows(prod, *first, &prod);
  }
  *out = prod;
  return overflow;
}

template <typename Container>
bool safe_multiplies_u64(const Container& c, uint64_t* out) {
  return safe_multiplies_u64(c.begin(), c.end(), out);
}

} // namespace c10
