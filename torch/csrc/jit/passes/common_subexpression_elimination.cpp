#include <torch/csrc/jit/ir.h>

#include <algorithm>
#include <unordered_map>

#include <torch/csrc/jit/assertions.h>
#include <torch/csrc/jit/interned_strings.h>
#include <torch/csrc/jit/node_hashing.h>
#include <torch/csrc/jit/passes/alias_analysis.h>
#include <torch/csrc/jit/passes/common_subexpression_elimination.h>
#include <torch/csrc/utils/functional.h>
#include <torch/csrc/utils/hash.h>

namespace torch {
namespace jit {
namespace {
// The function implements common subexpression elimination.
// Since the nodes are visited in topological order, one pass is enough.
void EliminateCommonSubexpression(
    Block* block,
    const AliasDb& aliasDb,
    std::function<Node*(Node*)> parent_lookup_fn) {
  std::unordered_set<Node*, HashNode, EqualNode> subexprs;
  for (auto it = block->nodes().begin(); it != block->nodes().end(); ++it) {
    auto node = *it;
    if (node->hasSideEffects() || node->isNondeterministic() ||
        aliasDb.hasWriters(node)) {
      // Do NOT have enough information to do CSE on these nodes.
      continue;
    }

    if (!node->blocks().empty()) {
      // Traverse sub-blocks.
      for (auto block : node->blocks()) {
        EliminateCommonSubexpression(block, aliasDb, [&](Node* n) {
          auto existing = subexprs.find(n);
          if (existing != subexprs.end()) {
            return *existing;
          }

          return parent_lookup_fn(n);
        });
      }

      continue;
    }

    // Check for CSE opportunities in the parent block.
    auto parent_lookup = parent_lookup_fn(node);
    if (parent_lookup) {
      node->replaceAllUsesWith(parent_lookup);
      it.destroyCurrent();
      continue;
    }

    // Check whether the same subexpression already exists.
    auto subit = subexprs.insert(node);
    if (!subit.second) {
      // Subexpression exists, replace the uses of node, and destroy it.
      auto existing = *subit.first;
      node->replaceAllUsesWith(existing);
      // Destroy the node.
      it.destroyCurrent();
    }
  }
}
} // namespace

void EliminateCommonSubexpression(std::shared_ptr<Graph>& graph) {
  const auto aliasDb = AliasAnalysis(graph);
  EliminateCommonSubexpression(
      graph->block(), aliasDb, [](Node*) { return nullptr; });
}
} // namespace jit
} // namespace torch
