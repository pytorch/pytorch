#pragma once

#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/passes/onnx/constant_map.h>
#include <torch/csrc/jit/passes/onnx/helper.h>
#include <torch/csrc/onnx/onnx.h>
#include <unordered_map>

namespace torch {
namespace jit {

TORCH_API std::shared_ptr<Graph> ToONNX(
    std::shared_ptr<Graph>& state,
    ::torch::onnx::OperatorExportTypes operator_export_type);
TORCH_API std::unordered_map<Value*, Value*> BlockToONNX(
    Block* old_block,
    Block* new_block,
    ::torch::onnx::OperatorExportTypes operator_export_type,
    std::unordered_map<Value*, Value*>& env,
    bool is_sub_block = false);
TORCH_API void NodeToONNX(
    Node* old_node,
    Block* new_block,
    ::torch::onnx::OperatorExportTypes operator_export_type,
    std::unordered_map<Value*, Value*>& env);
TORCH_API void RemovePrintOps(std::shared_ptr<Graph>& graph);
TORCH_API void PreprocessCaffe2Ops(std::shared_ptr<Graph>& graph);

} // namespace jit
} // namespace torch
