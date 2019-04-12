#pragma once
#include <functional>
#include <memory>
#include <string>

#include <torch/csrc/jit/ir.h>
#include <torch/csrc/jit/script/error_report.h>
#include <torch/csrc/jit/script/module.h>
#include <torch/csrc/jit/script/sugared_value.h>
#include <torch/csrc/jit/script/tree_views.h>

namespace torch {
namespace jit {
namespace script {

inline std::shared_ptr<SugaredValue> nativeResolver(
    const std::string& name,
    Function& m,
    const SourceRange& loc) {
  if (name == "torch") {
    return std::make_shared<BuiltinModule>("aten");
  }
  return nullptr;
}


TORCH_API void lambdaLiftFork(Node* fork_node);

} // namespace script
} // namespace jit
} // namespace torch
