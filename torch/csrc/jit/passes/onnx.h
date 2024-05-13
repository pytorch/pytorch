#pragma once

#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/onnx/onnx.h>
#include <torch/csrc/utils/pybind.h>

namespace torch {
namespace jit {

TORCH_API std::shared_ptr<Graph> ToONNX(
    std::shared_ptr<Graph>& state,
    ::torch::onnx::OperatorExportTypes operator_export_type);
TORCH_API py::dict BlockToONNX(
    Block* old_block,
    Block* new_block,
    ::torch::onnx::OperatorExportTypes operator_export_type,
    py::dict& env,
    py::set& values_in_env,
    bool is_sub_block = false);
TORCH_API void NodeToONNX(
    Node* old_node,
    Block* new_block,
    ::torch::onnx::OperatorExportTypes operator_export_type,
    py::dict& env,
    py::set& values_in_env);
TORCH_API void RemovePrintOps(std::shared_ptr<Graph>& graph);
TORCH_API void PreprocessCaffe2Ops(std::shared_ptr<Graph>& graph);

} // namespace jit
} // namespace torch
