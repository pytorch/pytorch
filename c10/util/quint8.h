#pragma once
#include <cstdint>

namespace c10 {

/**
 * qint8 is for signed 8 bit quantized Tensors
 */
struct alignas(1) quint8 {
  using underlying = uint8_t;
  uint8_t val_;
  explicit quint8(uint8_t val) : val_(val) {}
};

} // namespace c10
