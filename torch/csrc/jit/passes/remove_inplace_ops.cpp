#include <torch/csrc/jit/passes/remove_inplace_ops.h>

namespace torch {
namespace jit {
namespace {
static const std::unordered_map<NodeKind, NodeKind> inPlaceToOutOfPlace = {
    {aten::add_, aten::add},
    {aten::sub_, aten::sub},
    {aten::div_, aten::div},
    {aten::mul_, aten::mul},
    {aten::zero_, aten::zeros_like},
    {aten::fill_, aten::full_like}
    };

// This is a horrible no good awful hack to "fill in" the TensorOptions
// arguments of zeros_like and full_like so that the defaults are filled
// in.  Ugh.  Would be better to just run the frontend to get the correct
// arity here.
static const std::unordered_map<NodeKind, int> expectedInputCount = {
    {aten::zero_, 6},
    {aten::fill_, 7}};

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
      newNode->setScope(node->scope());
      // copy inputs
      for (auto input : node->inputs()) {
        newNode->addInput(input);
      }

      int additionalInputCount = 0;
      if (expectedInputCount.find(node->kind()) != expectedInputCount.end()) {
        additionalInputCount = expectedInputCount.at(node->kind()) - static_cast<int>(newNode->inputs().size());
      }

      for (int i = 0; i < additionalInputCount; ++i) {
        auto noneNode = graph->createNone();
        noneNode->insertBefore(newNode);
        newNode->addInput(noneNode->output());
      }

      // Create a new output node and replace all uses of self with it
      newNode->output()->copyMetadata(node->output());
      node->replaceAllUsesWith(newNode);
      node->inputs()[0]->replaceAllUsesAfterNodeWith(newNode, newNode->output());
      node->destroy();
    }
  }
}
} // namespace

void RemoveInplaceOps(const std::shared_ptr<Graph>& graph) {
  RemoveInplaceOps(graph->block());
}
} // namespace jit
} // namespace torch
