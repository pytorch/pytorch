#pragma once

#include <torch/csrc/jit/ir/ir.h>

namespace torch {
namespace jit {

TORCH_API bool canFuseOnCPULegacy();
TORCH_API void overrideCanFuseOnCPULegacy(bool value);

// NB: Be sure to run DCE before fusion, because dead instructions
// can prevent fusion opportunities from being exploited.
// On Windows will noop, NYI
TORCH_API void FuseGraph(
    std::shared_ptr<Graph>& graph,
    bool strict_fuser_check = false);

// \brief Custom fusion pass using a node-level callback to
// determine the inclusion of nodes in a subgraph.
//
// This helper omits aliased inputs and fusion across control flow
// boundaries.
//
// \arg graph The graph to be modified in-place
// \arg is_fusable A callback run on each fusable node in the graph.
// \arg kind The label given to the resultant fused subgraph
// \arg arg_limit The maximum number of args the resultant fused subgraph
//                should have.  Note: This will likely develop into a general
//                post condition on the fused subgraph.
TORCH_API void CustomFuseGraph(
    std::shared_ptr<Graph>& graph,
    const std::function<bool(Node*)>& is_fusable,
    Symbol kind,
    size_t arg_limit = std::numeric_limits<size_t>::max());

} // namespace jit
} // namespace torch
