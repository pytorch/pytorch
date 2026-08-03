#pragma once
#include <torch/csrc/Export.h>
#include <memory>
#include <ostream>
#include <string>

namespace torch::jit::onnx {

TORCH_API bool is_log_enabled();

TORCH_API void set_log_enabled(bool enabled);

TORCH_API void set_log_output_stream(std::shared_ptr<std::ostream> out_stream);

TORCH_API std::ostream& _get_log_output_stream();

#define ONNX_LOG(...)                            \
  if (::torch::jit::onnx::is_log_enabled()) {    \
    ::torch::jit::onnx::_get_log_output_stream() \
        << ::c10::str(__VA_ARGS__) << std::endl; \
  }

} // namespace torch::jit::onnx
