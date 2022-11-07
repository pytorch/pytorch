#pragma once
#include <cstdint>

#include <c10/macros/Macros.h>

namespace c10 {

/**
 * bit16 is an uninterpreted dtype of a tensor with 16 bits, without any
 * semantics defined.
 */
struct alignas(1) bit16 {
  uint16_t val_;
  bit16() = default;
  C10_HOST_DEVICE explicit bit16(uint16_t val) : val_(val) {}
};

} // namespace c10
