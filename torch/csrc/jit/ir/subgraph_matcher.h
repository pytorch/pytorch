#pragma once

#include <torch/csrc/jit/ir.h>

#include <unordered_map>
#include <vector>

namespace torch {
namespace jit {

/**
 * \brief A structure describing a match of a pattern in a graph.
 *
 * The structure contains an anchor node, from which the match was found, and
 * match-maps for nodes and values. A match-map specifies correspondance between
 * nodes in the pattern graph (match-map keys) with nodes in the actual graph
 * (match-map values). We keep such maps for both nodes and values.
 */
struct Match {
  Node* anchor;
  std::unordered_map<const Node*, Node*> nodes_map;
  std::unordered_map<const Value*, Value*> values_map;
};

/**
 * \brief Find all matches of a \p PATTERN in a \p GRAPH.
 *
 * The function returns a vector of match-descriptors (see description of
 * `struct Match`).
 *
 * Matching rules:
 *  - Pattern graph must contain a single block.
 *  - Matched subgraphs do not span across different blocks.
 *  - No uses outside the match are allowed, except for Param and Return nodes.
 *  Basically, we're matching hammocks, not arbitrary subgraphs.
 *  - Pattern graph must return only one value (i.e. it must have a single
 *  node leading to return).
 *  - Nodes that are not used in computation of the return value in the pattern
 * graph are ignored during matching (IOW, we're essentially performing DCE on
 * the pattern).
 *  - Pattern graph nodes cannot alias. TODO: the check not implemented yet.
 *  - Aliasing nodes in the graph can not consitute a match (i.e. in all found
 * matches no nodes in the subgraph alias with each other). TODO: the check not
 * implemented yet.
 *  - The matcher will not mutate either the pattern graph or the matched graph,
 * but the latter is taken as non-const so that Match may contain non-const
 * pointers.  This enables clients of this API to use Match to drive mutations.
 */
std::vector<Match> TORCH_API
findPatternMatches(const Graph& pattern, Graph& graph);

} // namespace jit
} // namespace torch
