#pragma once
#include <cstdint>

namespace c10 {

/**
 * This is the data type for quantized Tensors. Right now we only have
 * qint8 which is for 8 bit Tensors, we might have 4 bit, 2 bit or 1 bit
 * data types in the future.
 */
struct alignas(1) qint8 {
  uint8_t val_;
  /* implicit */ qint8(uint8_t val=0) : val_(val) {}

};

} // namespace c10
