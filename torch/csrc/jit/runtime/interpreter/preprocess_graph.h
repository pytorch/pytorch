#pragma once

#include <memory>
#include <unordered_map>

#include <torch/csrc/jit/ir/ir.h>

namespace torch {
namespace jit {
namespace interpreter {

// pre-processing that happens once per graph
struct PreprocessGraph {
  explicit PreprocessGraph(Graph& g);

  // Outputs of the preprocessing:
  std::shared_ptr<Graph> graph;
  std::unordered_map<Node*, bool> can_emit_inline;
};

} // namespace interpreter
} // namespace jit
} // namespace torch
