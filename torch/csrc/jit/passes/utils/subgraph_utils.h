#pragma once

#include <torch/csrc/WindowsTorchApiMacro.h>
#include <torch/csrc/jit/ir/ir.h>

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
// An optional argument 'vmap' could be used to retrieve value mappings
// Values will be mapped to their new subgraph values
TORCH_API Node* createSingletonSubgraph(Node* n, Symbol subgraphKind);
TORCH_API Node* createSingletonSubgraph(
    Node* n,
    Symbol subgraphKind,
    std::unordered_map<Value*, Value*>& vmap);

// Merge a node into a subgraph node. If `toMerge` is also a subgraph, the
// subgraphs are merged.
// `toMerge` is destroyed.
// An optional argument 'vmap' could be used to retrieve value mappings.
// Values will be mapped to their new subgraph values
TORCH_API void mergeNodeIntoSubgraph(Node* toMerge, Node* subgraphNode);
TORCH_API void mergeNodeIntoSubgraph(
    Node* toMerge,
    Node* subgraphNode,
    std::unordered_map<Value*, Value*>& vmap);

// Move nodes from a subgraph node to the outer graph.
// `subgraphNode` is destroyed.
// An optional argument 'vmap' could be used to retrieve value mappings.
TORCH_API void unmergeSubgraph(Node* subgraphNode);
TORCH_API void unmergeSubgraph(
    Node* subgraphNode,
    std::unordered_map<Value*, Value*>& vmap);

// Convenience function
std::shared_ptr<Graph> getSubgraph(Node* n);

} // namespace SubgraphUtils
} // namespace jit
} // namespace torch
