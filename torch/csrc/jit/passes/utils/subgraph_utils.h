#pragma once

#include <torch/csrc/Export.h>
#include <torch/csrc/jit/ir/alias_analysis.h>
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
TORCH_API Node* createSingletonSubgraph(Node* n, Symbol subgraphKind);

// Creates a new subgraph that only contains `n`, amd updates the new outputs
// of the subgraph to have the aliasing properties of the original `n` outputs
TORCH_API Node* createSingletonSubgraphAndUpdateAliasing(
    Node* to_merge,
    Symbol subgraphKind,
    AliasDb& db);

// Merge a node into a subgraph node. If `toMerge` is also a subgraph, the
// subgraphs are merged.
// If `destroyNode` is true `toMerge` is destroyed.
// An optional argument 'vmap' could be used to retrieve value mappings.
// Values will be mapped to their new subgraph values
TORCH_API void mergeNodeIntoSubgraph(
    Node* toMerge,
    Node* subgraphNode,
    bool destroyNode = true);

// Merges a node into a subgraph node, and updates the new outputs of the
// subgraph to have the aliasing properties of the corresponding `to_merge`
// outputs
TORCH_API void mergeNodeIntoSubgraphAndUpdateAliasing(
    Node* to_merge,
    Node* subgraphNode,
    AliasDb& db);

TORCH_API std::vector<Node*> unmergeAliasedOutputs(
    Node* subgraphNode,
    AliasDb& db);

// Move nodes from a subgraph node to the outer graph.
// `subgraphNode` is destroyed.
TORCH_API void unmergeSubgraph(Node* subgraphNode);

// Move `node_to_unmerge` and its descendants after `subgraphNode`
// promotes any dependencies of `node_to_unmerge` to subgraphNode outputs
TORCH_API void unmergeNode(Node* node_to_unmerge, Node* subgraphNode);

TORCH_API bool unmergeOutputsAlisingInputs(Node* subgraphNode);

TORCH_API bool unmergeAliasedOutputs(Node* subgraphNode);

// Convenience function
std::shared_ptr<Graph> getSubgraph(Node* n);

TORCH_API std::string generateNameForGraph(
    const std::shared_ptr<Graph>& graph,
    size_t maxlen = 40,
    const std::string& prefix = "fused");

} // namespace SubgraphUtils
} // namespace jit
} // namespace torch
