#pragma once
#include <cstdint>

#include <c10/macros/Macros.h>

namespace c10 {

/**
 * bits8 is an uninterpreted dtype of a tensor with 8 bits, without any
 * semantics defined.
 */
struct alignas(1) bits8 {
  uint8_t val_;
  bits8() = default;
  C10_HOST_DEVICE explicit bits8(uint8_t val) : val_(val) {}
};

} // namespace c10
