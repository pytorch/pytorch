#pragma once
#include <cstdint>
#include <limits>
#include <math.h>

#include <c10/macros/Macros.h>

namespace c10 {

namespace detail {

// the kernels below are copy-pasted from fbgemm_gpu:
// https://github.com/pytorch/FBGEMM/blob/277677039bae25b2570a73013b03bfaa9d2a523e/fbgemm_gpu/include/fbgemm_gpu/quantize_ops_utils.h
// TODO(future): switch fbgemm_gpu to use these

// reference conversion kernels between float32 and float8
using fint32 = union fint32 {
  uint32_t I;
  float F;
};

// TODO: add a flag later to control whether underflow
// flushes to 0 or clips to smallest denorm number.
inline C10_HOST_DEVICE uint8_t float_to_hfp8(float val_fp, int ebits, int exponent_bias, float max_pos) {
  int mbits = 7 - ebits;
  fint32 val_out, bouncer, smallest_normal;

  val_out.F = val_fp;
  uint32_t sign_bit = val_out.I & 0x80000000;
  val_out.I = val_out.I & 0x7FFFFFFF;
  val_out.F = fminf(val_out.F, max_pos);

  smallest_normal.I = (127 - exponent_bias + 1)
      << 23; // smallest hfp8 normal number in FP32
  // I don't know if the input "min_pos" is the smallest denormalized number
  // or the smallest normalized number. The test below needs to be done with
  // the smallest normal number, which is the numerical value 2^(1-bias)

  // The conversion for denormalized values are slightly different. HFP8 is so
  // low precision that gradual underflow is probably crucial
  if (val_out.F >= smallest_normal.F) {
    // Use round to nearest even. We make use of the standard rounding mechanism
    // in FP32 rather than rounding the mantissa and handling tie-to-even and
    // incrementing exponent We want to round of 23-mbits of the FP32 value
    // val_in This can be done by adding a power of 2 exactly 23-mbits larger
    // than the exponent of val_in This forces val_in to be moved to the right
    // and rounding exact at the location corresponding to having mbits of
    // explicit mantissa left
    bouncer.I = (val_out.I & 0xFF800000) + ((23 - mbits) << 23);
    val_out.F = (bouncer.F + val_out.F) - bouncer.F;
    // adding the bouncer rounds off bits, and subtracting bouncer
    // leaves the desired value, albeit in FP32 encoding
    // All we need is to change the exponent encoding to using "bias"
    val_out.I = uint32_t(val_out.I - ((127 - exponent_bias) << 23))
        << (8 - ebits);
    val_out.I =
        ((val_out.I | sign_bit) >>
         24); // the 8 lsbs is the desired HFP8 encoding

  } else {
    // When the value is in the denormal range, IEEE numbers essentially becomes
    // a fixed point number. The lsb is the smallest non-zero number
    // 2^(1-bias-mbits) Hence, we define the bouncer so that its lsb is this
    // smallest non-zero number Adding the input to this bouncer forces rounding
    // to occur appropriately Also, in this situation, after adding the bouncer,
    // the 8 least significant bits of the sum is already the HFP8 encoding of
    // the desired result. Just need to restore the sign bit
    bouncer.I = (127 + (23 + (1 - exponent_bias - mbits))) << 23;
    val_out.F = bouncer.F + val_out.F;
    val_out.I = val_out.I | (sign_bit >> 24);
    ;
  }

  uint8_t bfp8_val = val_out.I; // get the 8 lsbs
  return bfp8_val;
}

inline C10_HOST_DEVICE float hfp8_to_float(uint8_t hfp8_val, int ebits, int exponent_bias) {
  fint32 val_out, sign, multiplier;

  sign.I = (hfp8_val & 0x80) << 24;
  val_out.I = (hfp8_val & 0x7F) << (24 - (8 - ebits));
  // so that the mantissa bits start at the mantissa bit positions of FP32
  // encoding

  // Let the hfp8 mantissa bits correspond to the value frac, 0 <= frac < 1
  // So if the hfp8 value is a normal number, it's value is 2^e x (1+frac)
  // where e is its (true, unbiased) exponent
  // If the hfp8 value is denormal, the value is 2^(1-bias) x frac

  // However, the bit pattern in the 8-bit exponent field of val_out.F
  // is bias+e when hfp8 is normal, and 0 when hfp8 is subnormal.
  // So, as an FP32 value, when hfp8 is normal, val_out.F represents the value
  // of 2^(bias+e-127) * (1+frac)
  // And when hfp8 is subnormal, val_out.F is also subnormal, and represents the
  // value of 2^(-126) * frac In either case, val_out.F corresponds to
  // 2^(bias-127) * (value of hfp8 input) Thus, if we multiply val_out.F by
  // 2^(127-bias), we obtain the hfp8 value as an FP32 number

  multiplier.I = (127 + (127 - exponent_bias))
      << 23; // multiplier.F is 2^(127-bias)
  val_out.F *= multiplier.F;
  val_out.I |= sign.I;
  return val_out.F;
}

// constants for float8_e4m3fn
const float E4M3FN_MAX_POS = 448.f;
const int E4M3FN_EBITS = 4;
const int E4M3FN_EXPONENT_BIAS = 7;

} // namespace detail

/**
 * float8_e4m3fn is the fp8 dtype from https://arxiv.org/abs/2209.05433
 * with a 4 bit exponent and a 3 bit mantissa. The exponent bias
 * is assumed to be 7. The `fn` suffix is to signify that only finite and NaN
 * values are supported.
 */
struct alignas(1) float8_e4m3fn {
  uint8_t val_;
  float8_e4m3fn() = default;

  struct from_bits_t {};
  static constexpr C10_HOST_DEVICE from_bits_t from_bits() {
    return from_bits_t();
  }

  constexpr C10_HOST_DEVICE float8_e4m3fn(uint8_t bits, from_bits_t) : val_(bits) {};
  inline C10_HOST_DEVICE float8_e4m3fn(float value) {
    val_ = detail::float_to_hfp8(
      value, detail::E4M3FN_EBITS, detail::E4M3FN_EXPONENT_BIAS, detail::E4M3FN_MAX_POS);
  };
  inline C10_HOST_DEVICE operator float() const {
    return detail::hfp8_to_float(
      val_, detail::E4M3FN_EBITS, detail::E4M3FN_EXPONENT_BIAS);
  };
};

// TODO(before land): add float8_e5m2

} // namespace c10


namespace std {

template <>
class numeric_limits<c10::float8_e4m3fn> {
  public:
    static constexpr bool is_signed = true;
    // e4m3_fn does not support infinity
    static constexpr bool has_infinity = false;
    static constexpr bool has_quiet_NaN = false;
    // 3 digits in mantissa + leading digit
    static constexpr int digits = 4;
    // lowest representable number is -448 (0.1111.110)
    static constexpr c10::float8_e4m3fn lowest() {
      return at::float8_e4m3fn(0x7E, at::float8_e4m3fn::from_bits());
    }
    // highest representable number is 448 (1.1111.110)
    static constexpr c10::float8_e4m3fn max() {
      return at::float8_e4m3fn(0xFE, at::float8_e4m3fn::from_bits());
    }
};

} // namespace std
