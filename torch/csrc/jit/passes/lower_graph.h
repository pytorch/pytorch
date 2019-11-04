#pragma once

#include <torch/csrc/jit/ir.h>

namespace torch {
namespace jit {

using ModulePtr = c10::intrusive_ptr<c10::ivalue::Object>;

// Given a graph with of a method which first argument is %self, lower it to a
// graph where all attributes accesses are replaced with explicit inputs of the
// graph (rather than results of prim::GetAttr executed on %self).
//
// Returns a tuple (graph, parameters) where the last module.parameters.size()
// inputs to the graph are the trainable parameters used in this method. The
// remaining inputs are the true inputs to the function.
TORCH_API std::pair<std::shared_ptr<Graph>, std::vector<at::Tensor>> LowerGraph(
    Graph& graph,
    const ModulePtr& self);

} // namespace jit
} // namespace torch
