#pragma once
#include <cstdint>
#include <c10/macros/Macros.h>

namespace c10 {

/**
 * qint32 is for 32 bit quantized Tensors
 */
struct C10_API qint32 {
  using underlying = int32_t;
  int32_t val_;
  explicit qint32(int32_t val) : val_(val) {}
};

} // namespace c10
