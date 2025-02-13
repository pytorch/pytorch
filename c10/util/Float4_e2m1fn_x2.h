#pragma once
#include <cstdint>

#include <c10/macros/Macros.h>

/// Defines the Float4_e2m1fn_x2 type (4-bit floating-point, two elements packed
/// into one byte) Binary configuration :
///
///   0123:4567
///   seem:seem
///
/// This is the FP4 dtype from the OCP MX format spec
/// (https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf,
/// Section 5.3.3)

namespace c10 {

struct alignas(1) Float4_e2m1fn_x2 {
  uint8_t val_;
  Float4_e2m1fn_x2() = default;
  C10_HOST_DEVICE explicit Float4_e2m1fn_x2(uint8_t val) : val_(val) {}
};

} // namespace c10
