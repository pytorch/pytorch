#include "remove_inplace_ops.h"
#include <torch/csrc/jit/passes/dead_code_elimination.h>

namespace torch {
namespace jit {
namespace {
static const std::unordered_map<NodeKind, NodeKind> inPlaceToOutOfPlace = {
    {aten::add_, aten::add},
    {aten::sub_, aten::sub},
    {aten::div_, aten::div},
    {aten::mul_, aten::mul}};

bool isInplaceOp(const Node* node) {
  return inPlaceToOutOfPlace.count(node->kind()) != 0;
}

// Remove all in-place ops and replace them with out-of-place equivalents.
// e.g.
//   %foo = aten::add_(%foo, %n)
// becomes
//   %foo.2 = aten::add(%foo, %n)
//
// NOTE: this is NOT SAFE, since it assumes that the LHS is not aliased by
// another value. This is only to avoid breaking ONNX export; when alias
// analysis is done we can emit a warning if someone tries to export.
void RemoveInplaceOps(Block* block) {
  auto graph = block->owningGraph();
  auto it = block->nodes().begin();
  while (it != block->nodes().end()) {
    auto node = *it;
    ++it;
    for (auto block : node->blocks()) {
      RemoveInplaceOps(block);
    }

    if (isInplaceOp(node)) {
      // create a replacement out of place op
      auto newNode = graph->create(inPlaceToOutOfPlace.at(node->kind()));
      newNode->insertBefore(node);

      // copy inputs
      for (auto input : node->inputs()) {
        newNode->addInput(input);
      }

      // Create a new output node and replace all uses of self with it
      newNode->output()->copyMetadata(node->output());
      node->replaceAllUsesWith(newNode);
      node->destroy();
    }
  }
}
}

void RemoveInplaceOps(std::shared_ptr<Graph> graph) {
  RemoveInplaceOps(graph->block());
}
} // namespace jit
} // namespace torch
