#pragma once
#include <functional>
#include <memory>
#include <string>

#include <torch/csrc/jit/ir.h>
#include <torch/csrc/jit/script/error_report.h>
#include <torch/csrc/jit/script/module.h>
#include <torch/csrc/jit/script/resolver.h>
#include <torch/csrc/jit/script/sugared_value.h>
#include <torch/csrc/jit/script/tree_views.h>

namespace torch {
namespace jit {
namespace script {

TORCH_API void runCleanupPasses(std::shared_ptr<Graph>& to_clean);

TORCH_API bool meaningfulName(const std::string& name);
TORCH_API void lambdaLiftFork(Node* fork_node);

} // namespace script
} // namespace jit
} // namespace torch
