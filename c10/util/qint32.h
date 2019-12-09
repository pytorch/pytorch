#pragma once
#include <cstdint>

namespace c10 {

/**
 * qint32 is for signed 32 bit quantized Tensors
 */
struct alignas(4) qint32 {
  using underlying = int32_t;
  int32_t val_;
  explicit qint32(int32_t val) : val_(val) {}
};

} // namespace c10
