#pragma once
#include <cstdint>

#include <torch/headeronly/macros/Macros.h>

/// Defines the Float8_e8m0fnu_x4 type (vectorized 8-bit floating-point, four
/// elements packed). This is the E8M0 dtype from the OCP MX format spec
/// (https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf,
/// Section 5.4.1)

namespace c10 {

struct alignas(4) Float8_e8m0fnu_x4 {
  uint32_t val_;
  Float8_e8m0fnu_x4() = default;
  C10_HOST_DEVICE explicit Float8_e8m0fnu_x4(uint32_t val) : val_(val) {}
};

/// Comparison operators
inline C10_HOST_DEVICE bool operator==(
    const Float8_e8m0fnu_x4& a,
    const Float8_e8m0fnu_x4& b) {
  return a.val_ == b.val_;
}

inline C10_HOST_DEVICE bool operator!=(
    const Float8_e8m0fnu_x4& a,
    const Float8_e8m0fnu_x4& b) {
  return a.val_ != b.val_;
}

} // namespace c10

HIDDEN_NAMESPACE_BEGIN(torch, headeronly)
using c10::Float8_e8m0fnu_x4;
using c10::operator==;
using c10::operator!=;
HIDDEN_NAMESPACE_END(torch, headeronly)
