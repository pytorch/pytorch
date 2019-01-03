#include <torch/csrc/jit/interned_strings.h>
#include <torch/csrc/jit/ir.h>
#include <torch/csrc/jit/node_hashing.h>
#include <torch/csrc/jit/passes/constant_pooling.h>
#include <unordered_set>

namespace torch {
namespace jit {

namespace {

// Very similar to the common subexpression elimination pass
// Move all constants to the beginning of the graph, and deduplicate
void ConstantPooling(
    Block* block,
    std::unordered_set<Node*, HashNode, EqualNode>& constants) {
  for (auto it = block->nodes().begin(); it != block->nodes().end();) {
    auto node = *it;
    // node may be moved to a different block so advance iterator now
    ++it;
    if (!node->blocks().empty()) {
      // Traverse sub-blocks.
      for (auto block : node->blocks()) {
        ConstantPooling(block, constants);
      }
      continue;
    }

    if (node->kind() != prim::Constant) {
      continue;
    }

    auto first_node = node->owningGraph()->block()->nodes().front();
    if (node != first_node)
      node->moveBefore(first_node);

    // Check whether the same constant already exists.
    auto subit = constants.insert(node);
    if (!subit.second) {
      // constant exists, replace the uses of node, and destroy it.
      auto existing = *subit.first;
      node->replaceAllUsesWith(existing);
      node->destroy();
    }
  }
}

} // anonymous namespace

void ConstantPooling(const std::shared_ptr<Graph>& graph) {
  std::unordered_set<Node*, HashNode, EqualNode> constants;
  ConstantPooling(graph->block(), constants);
}

} // namespace jit
} // namespace torch
