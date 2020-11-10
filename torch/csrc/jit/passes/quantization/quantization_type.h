#pragma once
#include <ostream>

namespace torch {
namespace jit {

// Quantization type (dynamic quantization, static quantization).
// Should match the Python enum in quantize_jit.py
enum QuantType : uint8_t {
  DYNAMIC = 0,
  STATIC = 1,
  QAT = 2,
  WEIGHT_ONLY = 3,
  ACTIVATION_ONLY = 4
};

std::ostream& operator<<(std::ostream& os, QuantType t);

} // namespace jit
} // namespace torch
