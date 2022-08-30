#pragma once

#include <torch/csrc/jit/ir/ir.h>

namespace torch {
namespace jit {
namespace onnx {

namespace ONNXScopeName {

std::string createFullScopeName(
    const std::string& class_name,
    const std::string& variable_name);
std::string variableName(torch::jit::ScopePtr scope);
std::string variableNameFromRoot(
    torch::jit::ScopePtr scope,
    const std::string& layer_separator);
std::string className(torch::jit::ScopePtr scope);
std::string classNameFromRoot(
    torch::jit::ScopePtr scope,
    const std::string& layer_separator);
bool isCompatibleScope(torch::jit::ScopePtr scope);

} // namespace ONNXScopeName

TORCH_API void AssignScopedNamesForNodeAndValue(std::shared_ptr<Graph>& graph);

} // namespace onnx
} // namespace jit
} // namespace torch
