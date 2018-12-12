#pragma once

#include <torch/csrc/jit/graph_executor.h>
#include <torch/csrc/jit/ir.h>
#include <torch/csrc/jit/operator.h>

namespace torch {
namespace jit {

// Verify that alias annotations are correct. See impl for definition of
// "correct".
//
// This function expects a graph with a single op with `unqualifiedOpName`, plus
// the inputs that you would otherwise have passed to the graph executor.
TORCH_API void checkAliasAnnotation(
    std::shared_ptr<Graph> graph,
    std::vector<IValue> pythonInputs,
    const std::string& unqualifiedOpName);
} // namespace jit
} // namespace torch
