#pragma once

#include <torch/csrc/jit/ir/ir.h>

namespace torch {
namespace jit {
namespace onnx {

namespace ONNXScopeName {

std::string CreateFullScopeName(
    const std::string& class_name,
    const std::string& variable_name);
std::string VariableName(torch::jit::ScopePtr scope);
std::string VariableNameFromRoot(
    torch::jit::ScopePtr scope,
    const std::string& layer_separator);
std::string ClassName(torch::jit::ScopePtr scope);
std::string ClassNameFromRoot(
    torch::jit::ScopePtr scope,
    const std::string& layer_separator);
bool IsCompatibleScope(torch::jit::ScopePtr scope);

} // namespace ONNXScopeName

TORCH_API void AssignScopedNamesForNodeAndValue(std::shared_ptr<Graph>& graph);

} // namespace onnx
} // namespace jit
} // namespace torch
