#pragma once
#include <cstdint>

#include <torch/headeronly/macros/Macros.h>

/// Defines the Float6_e2m3fn_x4 type (6-bit floating-point, four elements
/// packed into three bytes). This is the FP6 E2M3 dtype from the OCP MX format
/// spec
/// (https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf,
/// Section 5.3.2).
///
/// The three bytes form a little-endian 24-bit word, and element i occupies
/// bits [6i, 6i + 6) of it. This is the order the AMD CDNA4 FP6 MFMA operands,
/// CUTLASS sub-byte packing, and the DLPack sub-byte rule use. From MSB to LSB:
///
///   original value             |  val3  :  val2  :  val1  :  val0
///   ==============================================================
///   bit index (MSB==23, LSB==0) | 23..18 : 17..12 : 11..6  :  5..0
///   sign/exponent/mantissa     | seemmm : seemmm : seemmm : seemmm
///
/// Since the element size is not a power of two, typed kernels must not be
/// instantiated for this type; byte-level ops view it as uint8[3] instead.

namespace c10 {

struct alignas(1) Float6_e2m3fn_x4 {
  uint8_t val_[3];
  Float6_e2m3fn_x4() = default;
  C10_HOST_DEVICE explicit Float6_e2m3fn_x4(uint8_t b0, uint8_t b1, uint8_t b2)
      : val_{b0, b1, b2} {}
};

/// Comparison operators
inline C10_HOST_DEVICE bool operator==(
    const Float6_e2m3fn_x4& a,
    const Float6_e2m3fn_x4& b) {
  return a.val_[0] == b.val_[0] && a.val_[1] == b.val_[1] &&
      a.val_[2] == b.val_[2];
}

inline C10_HOST_DEVICE bool operator!=(
    const Float6_e2m3fn_x4& a,
    const Float6_e2m3fn_x4& b) {
  return !(a == b);
}

} // namespace c10

HIDDEN_NAMESPACE_BEGIN(torch, headeronly)
using c10::Float6_e2m3fn_x4;
using c10::operator==;
using c10::operator!=;
HIDDEN_NAMESPACE_END(torch, headeronly)
