#pragma once
#include <cstdint>

#include <c10/macros/Macros.h>

namespace c10 {

/**
 * TODO(before land): write this docblock
 */
struct alignas(1) Float4_e2m1fn_x2 {
  uint8_t val_;
  Float4_e2m1fn_x2() = default;
  C10_HOST_DEVICE explicit Float4_e2m1fn_x2(uint8_t val) : val_(val) {}
};

} // namespace c10
