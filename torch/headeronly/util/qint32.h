#pragma once
#include <cstdint>

#include <torch/headeronly/macros/Macros.h>

namespace torch::headeronly {

/**
 * qint32 is for signed 32 bit quantized Tensors
 */
struct alignas(4) qint32 {
  using underlying = int32_t;
  int32_t val_;
  qint32() = default;
  C10_HOST_DEVICE explicit qint32(int32_t val) : val_(val) {}
};

} // namespace torch::headeronly

namespace c10 {
  using torch::headeronly::qint32;
} // namespace c10
