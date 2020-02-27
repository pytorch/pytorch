
copy: fbcode/caffe2/torch/csrc/jit/frontend/convert_to_ssa.h
copyrev: bbb226b2516a8353dbcba174a92269f0e643dd76

#pragma once
#include <functional>
#include <memory>
#include <string>

#include <torch/csrc/WindowsTorchApiMacro.h>
#include <torch/csrc/jit/ir/ir.h>

namespace torch {
namespace jit {
namespace script {

// Convert a graph with Loads & Stores into SSA form
TORCH_API void ConvertToSSA(std::shared_ptr<Graph>& graph);

} // namespace script
} // namespace jit
} // namespace torch
