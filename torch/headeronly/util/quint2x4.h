#pragma once
#include <cstdint>

#include <torch/headeronly/macros/Macros.h>

namespace c10 {

/**
 * quint2x4 is for un-signed 2 bit quantized Tensors that are packed to byte
 * boundary.
 */
struct alignas(1) quint2x4 {
  using underlying = uint8_t;
  uint8_t val_;
  quint2x4() = default;
  C10_HOST_DEVICE explicit quint2x4(uint8_t val) : val_(val) {}
};

} // namespace c10

namespace torch::headeronly {
using c10::quint2x4;
} // namespace torch::headeronly
