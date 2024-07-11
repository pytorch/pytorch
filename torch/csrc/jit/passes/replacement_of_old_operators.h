#pragma once

#include <torch/csrc/jit/ir/ir.h>

namespace torch {
namespace jit {

// Find the valid upgrader graph for the upgrader and cache the result
// for later lookups. Will error out if there is no valid upgrader graph
// provided for the upgrader name.
std::shared_ptr<Graph> getUpgraderGraph(const std::string& upgrader_name);

TORCH_API void ReplaceOldOperatorsWithUpgraders(std::shared_ptr<Graph> graph);

} // namespace jit
} // namespace torch
