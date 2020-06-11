#pragma once

namespace torch {
namespace jit {

// Quantization type (dynamic quantization, static quantization).
// Should match the Python enum in quantize_script.py
enum class QuantType { DYNAMIC, STATIC, QAT };

} // namespace jit
} // namespace torch
