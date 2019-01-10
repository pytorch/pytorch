#ifndef CAFFE2_UTILS_FIXED_DIVISOR_H_
#define CAFFE2_UTILS_FIXED_DIVISOR_H_

#include <cstdlib>
#include <stdint.h>

namespace caffe2 {

// Utility class for quickly calculating quotients and remainders for
// a known integer divisor
template <typename T>
class FixedDivisor {
};

// Works for any positive divisor, 1 to INT_MAX. One 64-bit
// multiplication and one 64-bit shift is used to calculate the
// result.
template <>
class FixedDivisor<int32_t> {
 public:
  FixedDivisor(int32_t d) : d_(d) {
    calcSignedMagic();
  }

  uint64_t getMagic() const {
    return magic_;
  }

  int getShift() const {
    return shift_;
  }

  /// Calculates `q = n / d`.
  inline int32_t div(int32_t n) const {
    // In lieu of a mulhi instruction being available, perform the
    // work in uint64
    uint64_t mul64 = magic_ * (uint64_t) n;
    return (int32_t) (mul64 >> shift_);
  }

  /// Calculates `r = n % d`.
  inline int32_t mod(int32_t n) const {
    return n - d_ * div(n);
  }

  /// Calculates `q = n / d` and `r = n % d` together.
  inline void divMod(int32_t n, int32_t& q, int32_t& r) const {
    const int32_t quotient = div(n);
    q = quotient;
    r = n - d_ * quotient;
  }

 private:
  /**
     Calculates magic multiplicative value and shift amount for
     calculating `q = n / d` for signed 32-bit integers.
     Implementation taken from Hacker's Delight section 10.
  */
  void calcSignedMagic() {
    if (d_ == 1) {
      magic_ = UINT64_C(0x1) << 32;
      shift_ = 32;
      return;
    }

    const uint32_t two31 = UINT32_C(0x80000000);
    uint32_t ad = std::abs(d_);
    uint32_t t = two31 + ((uint32_t) d_ >> 31);
    uint32_t anc = t - 1 - t % ad;   // Absolute value of nc.
    uint32_t p = 31;                 // Init. p.
    uint32_t q1 = two31 / anc;       // Init. q1 = 2**p/|nc|.
    uint32_t r1 = two31 - q1 * anc;  // Init. r1 = rem(2**p, |nc|).
    uint32_t q2 = two31 / ad;        // Init. q2 = 2**p/|d|.
    uint32_t r2 = two31 - q2 * ad;   // Init. r2 = rem(2**p, |d|).
    uint32_t delta = 0;

    do {
      p = p + 1;
      q1 = 2 * q1;         // Update q1 = 2**p/|nc|.
      r1 = 2 * r1;         // Update r1 = rem(2**p, |nc|).

      if (r1 >= anc) {     // (Must be an unsigned
        q1 = q1 + 1;       // comparison here).
        r1 = r1 - anc;
      }

      q2 = 2 * q2;         // Update q2 = 2**p/|d|.
      r2 = 2 * r2;         // Update r2 = rem(2**p, |d|).

      if (r2 >= ad) {      // (Must be an unsigned
        q2 = q2 + 1;       // comparison here).
        r2 = r2 - ad;
      }

      delta = ad - r2;
    } while (q1 < delta || (q1 == delta && r1 == 0));

    int32_t magic = q2 + 1;
    if (d_ < 0) {
      magic = -magic;
    }
    shift_ = p;
    magic_ = (uint64_t) (uint32_t) magic;
  }

  int32_t d_;
  uint64_t magic_;
  int shift_;
};

} // namespace caffe2

#endif // CAFFE2_UTILS_FIXED_DIVISOR_H_
