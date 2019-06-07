#pragma once

#include <torch/csrc/jit/ir.h>
#include <torch/csrc/onnx/onnx.h>

namespace torch {
namespace jit {

TORCH_API std::shared_ptr<Graph> ToONNX(
    std::shared_ptr<Graph>& state,
    ::torch::onnx::OperatorExportTypes operator_export_type);
TORCH_API void BlockToONNX(
    Block* old_block,
    Block* new_block,
    ::torch::onnx::OperatorExportTypes operator_export_type,
    std::unordered_map<Value*, Value*> env);

} // namespace jit
} // namespace torch
