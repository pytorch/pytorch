#pragma once

#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/WindowsTorchApiMacro.h>

namespace torch {
namespace jit {

// Utilities for dealing with nodes that contain subgraphs.
//
// They handle the complexity of editing inputs/outputs as you merge nodes in
// and out of subgraphs.
namespace SubgraphUtils {

// Create a new subgraph node that contains only `n`. The new subgraph will have
// `subgraphKind` as its type.
//
// `n` is destroyed.
//
// Returns the new subgraph node.
TORCH_API Node* createSingletonSubgraph(Node* n, Symbol subgraphKind);

// Merge a node into a subgraph node. If `toMerge` is also a subgraph, the
// subgraphs are merged.
// `toMerge` is destroyed.
TORCH_API void mergeNodeIntoSubgraph(Node* toMerge, Node* subgraphNode);

// Move nodes from a subgraph node to the outer graph.
// `subgraphNode` is destroyed.
TORCH_API void unmergeSubgraph(Node* subgraphNode);

// Convenience function
std::shared_ptr<Graph> getSubgraph(Node* n);

} // namespace SubgraphUtils
} // namespace jit
} // namespace torch
