#include <torch/csrc/jit/passes/common_subexpression_elimination.h>

#include <torch/csrc/jit/ir/alias_analysis.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/ir/node_hashing.h>
#include <torch/csrc/jit/jit_log.h>

#include <unordered_map>

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
    if (node->hasSideEffects()) {
      GRAPH_DEBUG("Node was skipped due to side effects:\n", *node);
      continue;
    }
    if (node->isNondeterministic()) {
      GRAPH_DEBUG("Node was skipped due to its non determinism:\n", *node);
      continue;
    }
    if (aliasDb.hasWriters(node)) {
      GRAPH_DEBUG("Node was skipped due to alias analysis result:\n", *node);
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
    auto g_out = node->owningGraph()->outputs();
    if (parent_lookup) {
      if (!aliasDb.safeToChangeAliasingRelationship(
              node->outputs(), parent_lookup->outputs())) {
        continue;
      }

      GRAPH_UPDATE("Replacing\n", *node, "with\n", *parent_lookup);
      node->replaceAllUsesWith(parent_lookup);
      it.destroyCurrent();
      continue;
    }

    // Check whether the same subexpression already exists.
    auto subit = subexprs.insert(node);
    if (!subit.second) {
      // Subexpression exists, replace the uses of node, and destroy it.
      auto existing = *subit.first;

      // don't introduce new aliasing among graph outputs
      if (aliasDb.mayContainAlias(
              node->outputs(), node->owningGraph()->outputs()) &&
          aliasDb.mayContainAlias(existing->outputs(), g_out)) {
        continue;
      }

      GRAPH_UPDATE("Replacing\n", *node, "with\n", *existing);
      node->replaceAllUsesWith(existing);
      // Destroy the node.
      it.destroyCurrent();
    }
  }
}
} // namespace

void EliminateCommonSubexpression(const std::shared_ptr<Graph>& graph) {
  AliasDb aliasDb(graph);
  GRAPH_DUMP("Before CSE", graph);
  EliminateCommonSubexpression(
      graph->block(), aliasDb, [](Node*) { return nullptr; });
}
} // namespace jit
} // namespace torch
