#pragma once
#include <cstdint>

#include <torch/headeronly/macros/Macros.h>

namespace c10 {

/**
 * quint4x2 is for un-signed 4 bit quantized Tensors that are packed to byte
 * boundary.
 */
struct alignas(1) quint4x2 {
  using underlying = uint8_t;
  uint8_t val_;
  quint4x2() = default;
  C10_HOST_DEVICE explicit quint4x2(uint8_t val) : val_(val) {}
};

} // namespace c10

namespace torch::headeronly {
using c10::quint4x2;
} // namespace torch::headeronly
