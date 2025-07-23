#pragma once
#include <cstdint>

#include <torch/headeronly/macros/Macros.h>

namespace torch::headeronly {

/**
 * quint8 is for unsigned 8 bit quantized Tensors
 */
struct alignas(1) quint8 {
  using underlying = uint8_t;
  uint8_t val_;
  quint8() = default;
  C10_HOST_DEVICE explicit quint8(uint8_t val) : val_(val) {}
};

} // namespace torch::headeronly

namespace c10 {
  using torch::headeronly::quint8;
} // namespace c10
