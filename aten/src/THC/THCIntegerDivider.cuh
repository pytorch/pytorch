#ifndef THC_INTEGER_DIVIDER_INC
#define THC_INTEGER_DIVIDER_INC

#include <assert.h>
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
#include <cuda_runtime.h>
#endif

// A utility class to implement integer division by multiplication, given a fixed
// divisor.
//
// WARNING: The fast divider algorithm is only implemented for unsigned int;
//          otherwise we default to plain integer division.  For unsigned int,
//          we further assume that the dividend is at most INT32_MAX.  Thus,
//          IntDivider must NOT be used for general integer division.
//
//          This reduced range is enough for our purpose, and it allows us to
//          slightly simplify the computation.
//
// (NOTE: Below, "2^k" denotes exponentiation, i.e., 1<<k.)
//
// For any N-bit unsigned integer d (> 0), we can find a "magic number" m (2^N
// <= m < 2^(N+1)) and shift s such that:
//
//    \floor(n / d) = \floor((m * n) / 2^(N+s)).
//
// Given such m and s, the integer division can be then implemented as:
//
//    let m' = m - 2^N  // 0 <= m' < 2^N
//
//    fast_integer_division(n):
//      // Multiply two N-bit unsigned integers: the result is a 2N-bit unsigned
//      // integer.  Then take the higher N bits.
//      t = (m' * n) >> N
//
//      // Here we use the fact that n is less than 2^(N-1): otherwise the value
//      // of (t + n) may not fit in an N-bit integer.
//      return (t + n) >> s
//
// Finding such a magic number is surprisingly easy:
//
//    s  = \ceil(\log_2 d)
//    m' = \floor(2^N * (2^s - d) / d) + 1  // Need 2N-bit integer arithmetic.
//
// See also:
//    - Division by Invariant Integers Using Multiplication,
//      Torbjörn Granlund and Peter L. Montgomery, 1994.
//
//    - http://www.hackersdelight.org/magic.htm
//
//    - http://ridiculousfish.com/blog/posts/labor-of-division-episode-i.html

// Result of div/mod operation stored together.
template <typename Value>
struct DivMod {
  Value div, mod;

  __host__ __device__ DivMod(Value div, Value mod) : div(div), mod(mod) { }
};

// Base case: we only have an implementation for uint32_t for now.  For
// everything else, we use plain division.
template <typename Value>
struct IntDivider {
  IntDivider() { }  // Dummy constructor for arrays.
  IntDivider(Value d) : divisor(d) { }

  __host__ __device__ inline Value div(Value n) const { return n / divisor; }
  __host__ __device__ inline Value mod(Value n) const { return n % divisor; }
  __host__ __device__ inline DivMod<Value> divmod(Value n) const {
    return DivMod<Value>(n / divisor, n % divisor);
  }

  Value divisor;
};

// Implement fast integer division.
template <>
struct IntDivider<unsigned int> {
  static_assert(sizeof(unsigned int) == 4, "Assumes 32-bit unsigned int.");

  IntDivider() { }  // Dummy constructor for arrays.

  IntDivider(unsigned int d) : divisor(d) {
    assert(divisor >= 1 && divisor <= INT32_MAX);

    // TODO: gcc/clang has __builtin_clz() but it's not portable.
    for (shift = 0; shift < 32; shift++) if ((1U << shift) >= divisor) break;

    uint64_t one = 1;
    uint64_t magic = ((one << 32) * ((one << shift) - divisor)) / divisor + 1;
    m1 = magic;
    assert(m1 > 0 && m1 == magic);  // m1 must fit in 32 bits.
  }

  __host__ __device__ inline unsigned int div(unsigned int n) const {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    // 't' is the higher 32-bits of unsigned 32-bit multiplication of 'n' and
    // 'm1'.
    unsigned int t = __umulhi(n, m1);
    return (t + n) >> shift;
#else
    // Using uint64_t so that the addition does not overflow.
    uint64_t t = ((uint64_t) n * m1) >> 32;
    return (t + n) >> shift;
#endif
  }

  __host__ __device__ inline unsigned int mod(unsigned int n) const {
    return n - div(n) * divisor;
  }

  __host__ __device__ inline DivMod<unsigned int> divmod(unsigned int n) const {
    unsigned int q = div(n);
    return DivMod<unsigned int>(q, n - q * divisor);
  }

  unsigned int divisor;  // d above.
  unsigned int m1;  // Magic number: m' above.
  unsigned int shift;  // Shift amounts.
};

#endif // THC_INTEGER_DIVIDER_INC
