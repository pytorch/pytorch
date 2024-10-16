#pragma once
#include <functional>
#include <memory>
#include <string>

#include <torch/csrc/jit/api/module.h>
#include <torch/csrc/jit/frontend/error_report.h>
#include <torch/csrc/jit/frontend/resolver.h>
#include <torch/csrc/jit/frontend/sugared_value.h>
#include <torch/csrc/jit/frontend/tree_views.h>
#include <torch/csrc/jit/ir/ir.h>

namespace torch::jit {

TORCH_API void runCleanupPasses(std::shared_ptr<Graph>& to_clean);

TORCH_API bool meaningfulName(const std::string& name);

} // namespace torch::jit
