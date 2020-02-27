
copy: fbcode/caffe2/torch/csrc/jit/runtime/jit_exception.cpp
copyrev: 9983ae6f81fe94da6112d5815fd879f6e830770e

#include <torch/csrc/jit/runtime/jit_exception.h>

namespace torch {
namespace jit {

JITException::JITException(const std::string& msg) : std::runtime_error(msg) {}

} // namespace jit
} // namespace torch
