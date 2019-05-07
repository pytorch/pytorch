#pragma once
#include <cstdint>
#include <c10/macros/Macros.h>

namespace c10 {

/**
 * This is the data type for quantized Tensors. Right now we only have
 * qint8 which is for 8 bit Tensors, and qint32 for 32 bit int Tensors,
 * we might have 4 bit, 2 bit or 1 bit data types in the future.
 */
struct C10_API qint8 {
  using underlying = uint8_t;
  uint8_t val_;
  explicit qint8(uint8_t val) : val_(val) {}
};

} // namespace c10
