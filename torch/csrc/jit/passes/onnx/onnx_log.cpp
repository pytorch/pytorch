#include <torch/csrc/jit/passes/onnx/onnx_log.h>
#include <iostream>

namespace torch {
namespace jit {

static bool onnx_log_enabled = false;
static std::ostream* out = &std::cout;

bool is_onnx_log_enabled() {
  return onnx_log_enabled;
}

void set_onnx_log_enabled(bool enabled) {
  onnx_log_enabled = enabled;
}

void set_onnx_log_output_stream(std::ostream* out_stream) {
  if (nullptr != out_stream) {
    out = out_stream;
  }
}

std::ostream& _get_onnx_log_output_stream() {
  return *out;
}

} // namespace jit
} // namespace torch
