#pragma once
#include <array>
#include <cstdint>

#include <torch/headeronly/macros/Macros.h>
#include <torch/headeronly/util/bit_cast.h>

/// Defines the Float4_e2m1fn_x2 type (4-bit floating-point, two elements packed
/// into one byte). This is the FP4 dtype from the OCP MX format spec
/// (https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf,
/// Section 5.3.3)
///
/// Given two high precision values val0 and val1, here is the
/// binary configuration of their packed representation, from MSB to LSB:
///
///   original value             | val1 : val0
///   ========================================
///   bit index (MSB==7, LSB==0) | 7654 : 3210
///   sign/exponent/mantissa     | seem : seem
///

namespace c10 {

namespace detail {

/// Convert a single fp32 value to its 4-bit e2m1 code (sign, 2 exponent, 1
/// mantissa), returned in the low nibble of a uint8_t. Values are rounded to
/// nearest even and clamped to the max representable magnitude (6.0); there is
/// no NaN/inf support. This is a scalar specialization (ebits=2, mbits=1) of
/// the branchless algorithm in
/// torch/testing/_internal/common_quantized.py::_f32_to_floatx_unpacked.
inline C10_HOST_DEVICE uint8_t fp4e2m1_from_fp32_value(float f) {
  constexpr uint32_t denorm_mask_int = 149u << 23;
  const float denorm_mask_float = c10::bit_cast<float>(denorm_mask_int);
  // (exp_bias - F32_EXP_BIAS) << MBITS_F32 + magic_adder == (-126 << 23) +
  // 2^21-1
  constexpr int32_t val_to_add = -1054867457;

  const uint32_t bits = c10::bit_cast<uint32_t>(f);
  const uint32_t sign = bits & 0x80000000u;
  const float x = c10::bit_cast<float>(bits ^ sign);

  uint8_t mag = 0;
  if (x >= 6.0f) {
    // saturate to max magnitude code
    mag = 7;
  } else if (x < 1.0f) {
    // denormal
    const uint32_t d = c10::bit_cast<uint32_t>(x + denorm_mask_float);
    mag = static_cast<uint8_t>(d - denorm_mask_int);
  } else {
    // normal: adjust exponent and round to nearest even
    int32_t nx = static_cast<int32_t>(bits ^ sign);
    const int32_t mant_odd = (nx >> 22) & 1;
    nx += val_to_add;
    nx += mant_odd;
    mag = static_cast<uint8_t>(nx >> 22);
  }
  const uint8_t sign_lp = static_cast<uint8_t>(sign >> 28) & 0x8;
  return (mag | sign_lp) & 0xF;
}

/// Convert a single 4-bit e2m1 code (in the low nibble of a uint8_t) back to
/// fp32. e2m1 has only 8 magnitudes, so the magnitude is a direct table lookup
/// (indexed by the 3 sign-less bits); the sign bit (bit 3) is applied
/// separately. This is the inverse of fp4e2m1_from_fp32_value.
inline C10_HOST_DEVICE float fp4e2m1_to_fp32_value(uint8_t code) {
  constexpr std::array<float, 8> mag_lut = {
      0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
  const float mag = mag_lut[code & 0x7];
  return (code & 0x8) ? -mag : mag;
}

} // namespace detail

struct alignas(1) Float4_e2m1fn_x2 {
  uint8_t val_;
  Float4_e2m1fn_x2() = default;
  C10_HOST_DEVICE explicit Float4_e2m1fn_x2(uint8_t val) : val_(val) {}
};

/// Comparison operators
inline C10_HOST_DEVICE bool operator==(
    const Float4_e2m1fn_x2& a,
    const Float4_e2m1fn_x2& b) {
  return a.val_ == b.val_;
}

inline C10_HOST_DEVICE bool operator!=(
    const Float4_e2m1fn_x2& a,
    const Float4_e2m1fn_x2& b) {
  return a.val_ != b.val_;
}

} // namespace c10

HIDDEN_NAMESPACE_BEGIN(torch, headeronly)
using c10::Float4_e2m1fn_x2;
using c10::operator==;
using c10::operator!=;
HIDDEN_NAMESPACE_END(torch, headeronly)
