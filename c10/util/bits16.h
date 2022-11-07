#pragma once
#include <cstdint>

#include <c10/macros/Macros.h>

namespace c10 {

/**
 * bits16 is an uninterpreted dtype of a tensor with 16 bits, without any
 * semantics defined.
 */
struct alignas(2) bits16 {
  uint16_t val_;
  bits16() = default;
  C10_HOST_DEVICE explicit bits16(uint16_t val) : val_(val) {}
};

} // namespace c10
