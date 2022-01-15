#pragma once
#include <torch/csrc/Export.h>
#include <ostream>
#include <string>

namespace torch {
namespace jit {

TORCH_API bool is_onnx_log_enabled();

TORCH_API void set_onnx_log_enabled(bool enabled);

TORCH_API void set_onnx_log_output_stream(std::ostream* out_stream);

TORCH_API std::ostream& _get_onnx_log_output_stream();

#define ONNX_LOG(...)                               \
  if (is_onnx_log_enabled()) {                 \
    ::torch::jit::_get_onnx_log_output_stream() \
        << ::c10::str(__VA_ARGS__) << std::endl;    \
  }

} // namespace jit
} // namespace torch
