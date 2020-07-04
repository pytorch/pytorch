#pragma once
#include <cstdint>

#include <c10/macros/Macros.h>

namespace c10 {

/**
 * qint32 is for signed 32 bit quantized Tensors
 */
struct alignas(4) qint32 {
  using underlying = int32_t;
  int32_t val_;
  qint32() = default;
  C10_HOST_DEVICE explicit qint32(int32_t val) : val_(val) {}
};

} // namespace c10
