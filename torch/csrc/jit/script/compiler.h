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

void runCleanupPasses(
    std::shared_ptr<Graph>& to_clean,
    bool convert_ssa = true);

bool meaningfulName(const std::string& name);

} // namespace script
} // namespace jit
} // namespace torch
