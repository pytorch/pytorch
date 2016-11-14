#ifndef CAFFE2_UTILS_FIXED_DIVISOR_H_
#define CAFFE2_UTILS_FIXED_DIVISOR_H_

#ifdef __ARM_NEON__
#include <arm_neon.h>
#endif
#include <cmath>

namespace caffe2 {

namespace detail {

inline uint32_t mulHi(uint32_t x, uint32_t y) {
  uint64_t v = (uint64_t) x * (uint64_t) y;
  return (uint32_t) (v >> 32);
}

}

// Utility class for quickly calculating quotients and remainders for
// a known integer divisor
template <typename T>
class FixedDivisor {
};

template <>
class FixedDivisor<int> {
 public:
  typedef int Type;

  FixedDivisor(int d) : d_(d) {
    calcSignedMagic();
  }

  /// Calculates `q = n / d`.
  inline int div(int n) const {
    return (int) (detail::mulHi(magic_, n) >> shift_);
  }

  /// Calculates `r = n % d`.
  inline int mod(int n) const {
    return n - d_ * div(n);
  }

  /// Calculates `q = n / d` and `r = n % d` together.
  inline void divMod(int n, int& q, int& r) const {
    const int quotient = div(n);
    q = quotient;
    r = n - d_ * quotient;
  }

#ifdef __ARM_NEON__
  inline void divModVector(int32x4_t n,
                           int32x4_t& q,
                           int32x4_t& r) const {
    int32x2_t loQ;
    int32x2_t loR;
    divModVector(vget_low_s32(n), loQ, loR);

    int32x2_t hiQ;
    int32x2_t hiR;
    divModVector(vget_high_s32(n), hiQ, hiR);

    q = vcombine_s32(loQ, hiQ);
    r = vcombine_s32(loR, hiR);
  }

  inline void divModVector(int32x2_t n,
                           int32x2_t& q,
                           int32x2_t& r) const {
    q = divVector(n);

    // r = n - d * q
    r = vsub_s32(n, vmul_s32(vdup_n_s32(d_), q));
  }

  // Calculates `q1 = v1 / d, q2 = v2 / d` using NEON
  inline int32x2_t divVector(int32x2_t v) const {
    uint32x2_t vUnsigned = vreinterpret_u32_s32(v);

    uint32x2_t resultUnsigned =
      vmovn_u64(
        vshlq_u64(
          vmull_u32(vUnsigned, vdup_n_u32(magic_)),
          vdupq_n_s64(-32 - shift_)));

    return vreinterpret_s32_u32(resultUnsigned);
  }
#endif

 private:
  /**
     Calculates magic multiplicative value and shift amount for
     calculating `q = n / d` for signed 32-bit integers.
     Implementation taken from Hacker's Delight section 10.
     `d` cannot be in [-1, 1].
  */
  void calcSignedMagic() {
    const unsigned int two31 = 0x80000000;

    unsigned int ad = std::abs(d_);
    unsigned int t = two31 + ((unsigned int) d_ >> 31);
    unsigned int anc = t - 1 - t % ad;   // Absolute value of nc.
    unsigned int p = 31;                 // Init. p.
    unsigned int q1 = two31 / anc;       // Init. q1 = 2**p/|nc|.
    unsigned int r1 = two31 - q1 * anc;  // Init. r1 = rem(2**p, |nc|).
    unsigned int q2 = two31 / ad;        // Init. q2 = 2**p/|d|.
    unsigned int r2 = two31 - q2 * ad;   // Init. r2 = rem(2**p, |d|).
    unsigned int delta = 0;

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

    magic_ = q2 + 1;
    if (d_ < 0) {
      magic_ = -magic_;
    }
    shift_ = p - 32;
  }

  int d_;
  int magic_;
  int shift_;
};

} // namespace caffe2

#endif // CAFFE2_UTILS_FIXED_DIVISOR_H_
