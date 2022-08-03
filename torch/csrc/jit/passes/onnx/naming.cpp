#include <torch/csrc/jit/passes/onnx/naming.h>
#include <torch/csrc/onnx/onnx.h>

namespace torch {
namespace jit {
namespace onnx {

namespace ONNXScopeName {

using NameFunc = std::string (*)(torch::jit::ScopePtr scope);

const std::string name_separator = "::";

namespace {

std::string NameFromRoot(
    torch::jit::ScopePtr scope,
    const std::string& layer_separator,
    NameFunc name_func) {
  std::string out = (*name_func)(scope);
  if (scope->isRoot()) {
    return out;
  }
  auto parent = scope->parent();
  while (!parent->isRoot()) {
    out = std::string((*name_func)(parent)).append(layer_separator).append(out);
    parent = parent->parent();
  }
  return out;
}

std::pair<std::string, std::string> ParseNameFromScope(
    torch::jit::ScopePtr scope) {
  std::string full_name = scope->name().toUnqualString();
  auto pos = full_name.find(name_separator);
  TORCH_CHECK(
      pos != std::string::npos,
      "Scope name (" + full_name + ") does not contain '" + name_separator +
          "'");
  return std::make_pair(full_name.substr(0, pos), full_name.substr(pos + 2));
}

} // namespace

std::string CreateFullScopeName(
    const std::string& class_name,
    const std::string& variable_name) {
  return std::string(class_name).append(name_separator).append(variable_name);
}

std::string VariableName(torch::jit::ScopePtr scope) {
  return ParseNameFromScope(scope).second;
}

std::string VariableNameFromRoot(
    torch::jit::ScopePtr scope,
    const std::string& layer_separator) {
  return NameFromRoot(scope, layer_separator, &VariableName);
}

std::string ClassName(torch::jit::ScopePtr scope) {
  return ParseNameFromScope(scope).first;
}

std::string ClassNameFromRoot(
    torch::jit::ScopePtr scope,
    const std::string& layer_separator) {
  return NameFromRoot(scope, layer_separator, &ClassName);
}

bool IsCompatibleScope(torch::jit::ScopePtr scope) {
  return !scope->isRoot() && !scope->isBlank() &&
      (std::string(scope->name().toUnqualString()).find(name_separator) !=
       std::string::npos);
}
} // namespace ONNXScopeName

} // namespace onnx
} // namespace jit
} // namespace torch
