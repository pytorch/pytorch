#pragma once
#include <cstdint>

#include <torch/headeronly/macros/Macros.h>

namespace c10 {

/**
 * This is the data type for quantized Tensors. Right now we only have
 * qint8 which is for 8 bit Tensors, and qint32 for 32 bit int Tensors,
 * we might have 4 bit, 2 bit or 1 bit data types in the future.
 */
struct alignas(1) qint8 {
  using underlying = int8_t;
  int8_t val_;
  qint8() = default;
  C10_HOST_DEVICE explicit qint8(int8_t val) : val_(val) {}
};

} // namespace c10

namespace torch::headeronly {
using c10::qint8;
} // namespace torch::headeronly
