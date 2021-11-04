#include <torch/csrc/jit/passes/constant_pooling.h>

#include <ATen/core/interned_strings.h>
#include <torch/csrc/jit/ir/alias_analysis.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/ir/node_hashing.h>
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
      auto existing = *subit.first;

      auto old_ivalue = toIValue(existing->output());
      auto new_ivalue = toIValue(node->output());

      // if both values are the same object, we do not need to worry about
      // changing the aliasing relationship
      bool same_identity =
          (old_ivalue && new_ivalue && (old_ivalue->is(new_ivalue)));

      if (!same_identity &&
          !aliasDb.safeToChangeAliasingRelationship(
              node->outputs(), existing->outputs())) {
        continue;
      }

      // constant exists, replace the uses of node, and destroy it.
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
