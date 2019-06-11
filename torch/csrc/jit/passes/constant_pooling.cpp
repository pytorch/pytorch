#include <torch/csrc/jit/passes/constant_pooling.h>
#include <ATen/core/interned_strings.h>
#include <torch/csrc/jit/ir.h>
#include <torch/csrc/jit/node_hashing.h>
#include <torch/csrc/jit/passes/alias_analysis.h>
#include <unordered_set>

namespace torch {
namespace jit {

namespace {

// Very similar to the common subexpression elimination pass
// Move all constants to the beginning of the graph, and deduplicate
void ConstantPooling(
    Block* block,
    std::unordered_set<Node*, HashNode, EqualNode>& constants,
    const AliasDb& aliasDb) {
  for (auto it = block->nodes().begin(); it != block->nodes().end();) {
    auto node = *it;
    // node may be moved to a different block so advance iterator now
    ++it;
    if (!node->blocks().empty()) {
      // Traverse sub-blocks.
      for (auto block : node->blocks()) {
        ConstantPooling(block, constants, aliasDb);
      }
      continue;
    }

    if (node->kind() != prim::Constant) {
      continue;
    }

    // Check whether the same constant already exists.
    auto subit = constants.insert(node);
    if (!subit.second) {
      // constant exists, replace the uses of node, and destroy it.
      auto existing = *subit.first;

      // since the graph outputs may be mutated after they are returned,
      // don't introduce new aliasing among graph outputs
      if (aliasDb.mayContainAlias(
              node->outputs(), node->owningGraph()->outputs()) &&
          aliasDb.mayContainAlias(
              existing->outputs(), node->owningGraph()->outputs())) {
        continue;
      }

      node->replaceAllUsesWith(existing);
      node->destroy();
      continue;
    }

    // Move the constant definition to the beginning of the graph.
    auto first_node = node->owningGraph()->block()->nodes().front();
    if (node != first_node)
      node->moveBefore(first_node);
  }
}
} // anonymous namespace

void ConstantPooling(const std::shared_ptr<Graph>& graph) {
  AliasDb aliasDb(graph);
  std::unordered_set<Node*, HashNode, EqualNode> constants;
  ConstantPooling(graph->block(), constants, aliasDb);
}
} // namespace jit
} // namespace torch
