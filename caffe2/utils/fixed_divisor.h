#ifndef CAFFE2_UTILS_FIXED_DIVISOR_H_
#define CAFFE2_UTILS_FIXED_DIVISOR_H_

#include <cstdint>
#include <cstdio>
#include <cstdlib>

// See Note [hip-clang differences to hcc]

#if defined(__CUDA_ARCH__) || defined(__HIP_ARCH__) || defined(__HIP__) || \
    (defined(__clang__) && defined(__CUDA__))
#define FIXED_DIVISOR_DECL inline __host__ __device__
#else
#define FIXED_DIVISOR_DECL inline
#endif

namespace caffe2 {

// Utility class for quickly calculating quotients and remainders for
// a known integer divisor
template <typename T>
class FixedDivisor {};

// Works for any positive divisor, 1 to INT_MAX. One 64-bit
// multiplication and one 64-bit shift is used to calculate the
// result.
template <>
class FixedDivisor<std::int32_t> {
 public:
  FixedDivisor() = default;

  explicit FixedDivisor(const std::int32_t d) : d_(d) {
#ifndef __HIP_PLATFORM_HCC__
    CalcSignedMagic();
#endif // __HIP_PLATFORM_HCC__
  }

  FIXED_DIVISOR_DECL std::int32_t d() const {
    return d_;
  }

#ifndef __HIP_PLATFORM_HCC__
  FIXED_DIVISOR_DECL std::uint64_t magic() const {
    return magic_;
  }

  FIXED_DIVISOR_DECL int shift() const {
    return shift_;
  }
#endif // __HIP_PLATFORM_HCC__

  /// Calculates `q = n / d`.
  FIXED_DIVISOR_DECL std::int32_t Div(const std::int32_t n) const {
#ifdef __HIP_PLATFORM_HCC__
    return n / d_;
#else // __HIP_PLATFORM_HCC__
    // In lieu of a mulhi instruction being available, perform the
    // work in uint64
    return (int32_t)((magic_ * (uint64_t)n) >> shift_);
#endif // __HIP_PLATFORM_HCC__
  }

  /// Calculates `r = n % d`.
  FIXED_DIVISOR_DECL std::int32_t Mod(const std::int32_t n) const {
    return n - d_ * Div(n);
  }

  /// Calculates `q = n / d` and `r = n % d` together.
  FIXED_DIVISOR_DECL void
  DivMod(const std::int32_t n, std::int32_t* q, int32_t* r) const {
    *q = Div(n);
    *r = n - d_ * *q;
  }

 private:
#ifndef __HIP_PLATFORM_HCC__
  // Calculates magic multiplicative value and shift amount for calculating `q =
  // n / d` for signed 32-bit integers.
  // Implementation taken from Hacker's Delight section 10.
  void CalcSignedMagic() {
    if (d_ == 1) {
      magic_ = UINT64_C(0x1) << 32;
      shift_ = 32;
      return;
    }

    const std::uint32_t two31 = UINT32_C(0x80000000);
    const std::uint32_t ad = std::abs(d_);
    const std::uint32_t t = two31 + ((uint32_t)d_ >> 31);
    const std::uint32_t anc = t - 1 - t % ad; // Absolute value of nc.
    std::uint32_t p = 31; // Init. p.
    std::uint32_t q1 = two31 / anc; // Init. q1 = 2**p/|nc|.
    std::uint32_t r1 = two31 - q1 * anc; // Init. r1 = rem(2**p, |nc|).
    std::uint32_t q2 = two31 / ad; // Init. q2 = 2**p/|d|.
    std::uint32_t r2 = two31 - q2 * ad; // Init. r2 = rem(2**p, |d|).
    std::uint32_t delta = 0;
    do {
      ++p;
      q1 <<= 1; // Update q1 = 2**p/|nc|.
      r1 <<= 1; // Update r1 = rem(2**p, |nc|).
      if (r1 >= anc) { // (Must be an unsigned
        ++q1; // comparison here).
        r1 -= anc;
      }
      q2 <<= 1; // Update q2 = 2**p/|d|.
      r2 <<= 1; // Update r2 = rem(2**p, |d|).
      if (r2 >= ad) { // (Must be an unsigned
        ++q2; // comparison here).
        r2 -= ad;
      }
      delta = ad - r2;
    } while (q1 < delta || (q1 == delta && r1 == 0));
    std::int32_t magic = q2 + 1;
    if (d_ < 0) {
      magic = -magic;
    }
    shift_ = p;
    magic_ = (std::uint64_t)(std::uint32_t)magic;
  }
#endif // __HIP_PLATFORM_HCC__

  std::int32_t d_ = 1;

#ifndef __HIP_PLATFORM_HCC__
  std::uint64_t magic_;
  int shift_;
#endif // __HIP_PLATFORM_HCC__
};

} // namespace caffe2

#endif // CAFFE2_UTILS_FIXED_DIVISOR_H_
