#pragma once
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
/// nearest even. fp4 has no inf/NaN encoding: inf and finite overflow saturate
/// to the max representable magnitude (6.0) preserving sign, while NaN clamps
/// to positive 6.0 to match the hardware fp4 convert intrinsic. This is a
/// scalar specialization (ebits=2, mbits=1) of the branchless algorithm in
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
  if (!(x < 6.0f)) {
    // saturation branch, for values >=6, inf and NaN
    // Note: `!(x < 6.0f)` instead of `(x >= 6.0f)` makes the comparison
    // also work for NaNs.
    mag = 7;
  } else if (x < 1.0f) {
    // denormal branch
    // x + 2^22
    const uint32_t d = c10::bit_cast<uint32_t>(x + denorm_mask_float);
    // strip the 2^22 bits out, leaving us with the target code bits
    mag = static_cast<uint8_t>(d - denorm_mask_int);
  } else {
    // normal branch
    int32_t nx = static_cast<int32_t>(bits ^ sign);
    const int32_t mant_odd = (nx >> 22) & 1;
    // rebias the exponent + round-to-nearest
    nx += val_to_add;
    // go from round-to-nearest to round-to-nearest-even
    nx += mant_odd;
    // shift right to the target code bits
    mag = static_cast<uint8_t>(nx >> 22);
  }
  // NaN clamps to positive max magnitude (matching the hardware fp4 convert
  // intrinsic); inf and finite overflow keep their sign. `f != f` is true only
  // for NaN.
  const uint8_t sign_lp =
      (f != f) ? 0 : (static_cast<uint8_t>(sign >> 28) & 0x8);
  return (mag | sign_lp) & 0xF;
}

/// Convert a single 4-bit e2m1 code (in the low nibble of a uint8_t) back to
/// fp32.
inline C10_HOST_DEVICE float fp4e2m1_to_fp32_value(uint8_t code) {
  // Decode the magnitude from the 3 low bits (the e2m1 code without its sign)
  // with a branch-free select tree. The tree walks the bits high-to-low, so the
  // leaves are in code order; magnitude by code is 0->0, 1->0.5, 2->1, 3->1.5,
  // 4->2, 5->3, 6->4, 7->6. This is deliberately not a runtime-indexed lookup
  // table (spills to slow local memory on CUDA) nor a switch/field arithmetic:
  // both measured ~3-54% slower on the reverse cast because they add register
  // pressure or serialized control flow, while these predicated selects on
  // immediate constants stay on the fast path.
  const uint8_t idx = code & 0x7;
  const float mag = (idx & 0x4) ? ((idx & 0x2) ? ((idx & 0x1) ? 6.0f : 4.0f)
                                               : ((idx & 0x1) ? 3.0f : 2.0f))
                                : ((idx & 0x2) ? ((idx & 0x1) ? 1.5f : 1.0f)
                                               : ((idx & 0x1) ? 0.5f : 0.0f));
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
