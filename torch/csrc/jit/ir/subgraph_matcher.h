#pragma once

#include <torch/csrc/jit/ir/ir.h>

#include <unordered_map>
#include <vector>

namespace torch {
namespace jit {

/**
 * \brief A structure describing a match of a pattern in a graph.
 *
 * The structure contains an anchor node, from which the match was found, and
 * match-maps for nodes and values. A match-map specifies the correspondance
 * between nodes in the pattern graph (match-map keys) with nodes in the actual
 * graph (match-map values). We keep such maps for both nodes and values.
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
 *  - The pattern graph must return only one value (i.e. it must have a single
 *  node leading to return).
 *  - Nodes that are not used in computation of the return value in the pattern
 * graph are ignored during matching (IOW, we're essentially performing DCE on
 * the pattern).
 *  - Pattern graph nodes cannot alias. TODO: the check not implemented yet.
 *  - Aliasing nodes in the graph cannot consitute a match (i.e. through all
 * found matches, no nodes in the subgraph alias with each other). TODO: check
 * not implemented yet.
 *  - The matcher will not mutate either the pattern graph or the matched graph.
 * The matched graph is taken as non-const so that Match may contain non-const
 * pointers.  This enables clients of this API to use Match to drive mutations.
 *
 * Note [Multi-output Patterns]
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 * Subgraph matcher provides limited support for multi-output patterns. With a
 * single output pattern, a single scan through the graph is sufficient to
 * find all the matches: given a starting node (an "anchor"), we can
 * deterministically check whether a pattern matches a subgraph corresponding to
 * this anchor node. For a general case of multi-output patterns, we would have
 * N anchors, which would result in M^N comparisons (M is the size of the
 * graph). Clearly this is computationally prohibitive.
 *
 * To overcome this, we impose some constraints on the multi-output patterns
 * that we accept. We require that checking whether the pattern matches a
 * subgraph would still be fully determined by a single node in the graph. To
 * achieve this, we designate the first output in the pattern as the "main"
 * output and assume that we can traverse up from this node to match the
 * entire pattern.
 *
 * Corrolary 1: the order of outputs in the pattern matters!
 * Corollary 2: patterns cannot contain any nodes not participating in the main
 * output computation.
 */
std::vector<Match> TORCH_API
findPatternMatches(const Graph& pattern, Graph& graph);

} // namespace jit
} // namespace torch
