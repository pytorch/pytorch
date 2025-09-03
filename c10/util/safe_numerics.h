#pragma once
#include <c10/macros/Macros.h>

#include <cstddef>
#include <cstdint>

// GCC has __builtin_mul_overflow from before it supported __has_builtin
#if defined(_MSC_VER) && !defined(__SYCL_DEVICE_ONLY__)
#define C10_HAS_BUILTIN_OVERFLOW() (0)
#include <intrin.h>
#include <cmath>
#include <limits>
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
#elif defined(_M_AMD64) && (_MSC_VER >= 1937)
  uint64_t high{0};
  return _mul_full_overflow_u64(a, b, out, &high);
#elif defined(_M_AMD64)
  uint64_t high{0};
  *out = _umul128(a, b, &high);
  return high; // overflow if high bits are non-zero
#elif defined(_M_ARM64)
  *out = a * b; // low 64 bits of the result
  return __umulh(a, b); // overflow if high bits are non-zero
#else
  static_assert(false, "Not implemented");
#endif
}

C10_ALWAYS_INLINE bool mul_overflows(int64_t a, int64_t b, int64_t* out) {
#if C10_HAS_BUILTIN_OVERFLOW()
  return __builtin_mul_overflow(a, b, out);
#elif defined(_M_AMD64) && (_MSC_VER >= 1937)
  return _mul_overflow_i64(a, b, out);
#elif defined(_M_AMD64)
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
#elif defined(_M_ARM64)
  // Idea: Perform an unsigned multiplication while safely
  // casting int64_t to uint64_t and vice-versa.
  int64_t int64_min{std::numeric_limits<int64_t>::min()}; // -2^63
  uint64_t int64_max{std::numeric_limits<int64_t>::max()}; // 2^63 - 1
  uint64_t abs_int64_min{int64_max + 1}; // 2^63
  uint64_t abs_a{(a == int64_min) ? abs_int64_min : std::abs(a)};
  uint64_t abs_b{(b == int64_min) ? abs_int64_min : std::abs(b)};
  uint64_t unsigned_result{0};
  if (mul_overflows(abs_a, abs_b, &unsigned_result)) {
    return true;
  }

  bool negative{(a < 0) == (b < 0)};
  if (!negative) {
    if (unsigned_result > int64_max) {
      return true;
    }
    *out = static_cast<int64_t>(unsigned_result);
    return false;
  }

  // safely negate the result
  if (unsigned_result > abs_int64_min) {
    return true;
  } else if (unsigned_result == abs_int64_min) {
    *out = int64_min;
    return false;
  } else {
    *out = -static_cast<int64_t>(unsigned_result);
    return false;
  }
#else
  static_assert(false, "Not implemented");
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
