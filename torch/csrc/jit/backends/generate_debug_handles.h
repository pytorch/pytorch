#pragma once
#include <ATen/core/jit_type.h>

#include <torch/csrc/jit/api/module.h>
#include <torch/csrc/jit/backends/backend_debug_handler.h>
#include <torch/csrc/jit/ir/ir.h>

namespace torch {
namespace jit {

using NodeToDebugHandle = std::unordered_map<Node*, DebugHandleType>;
using DebugHandleToDebugInfo = std::unordered_map<DebugHandleType, DelegateDebugInfoType>;

std::pair<NodeToDebugHandle, DebugHandleToDebugInfo> generate_debug_handles(
    const Module& mod, const std::vector<std::string>& method_names);

} // namespace jit
} // namespace torch
