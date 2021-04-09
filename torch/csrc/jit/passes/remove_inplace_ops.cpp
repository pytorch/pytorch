#include <torch/csrc/jit/passes/remove_inplace_ops.h>

namespace torch {
namespace jit {
namespace {
static const std::unordered_map<NodeKind, NodeKind> inPlaceToOutOfPlace = {
    {aten::add_, aten::add},
    {aten::sub_, aten::sub},
    {aten::div_, aten::div},
    {aten::mul_, aten::mul},
    {aten::masked_fill_, aten::masked_fill},
    {aten::zero_, aten::zeros_like},
    {aten::fill_, aten::full_like}};

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
        additionalInputCount = expectedInputCount.at(node->kind()) -
            static_cast<int>(newNode->inputs().size());
      }

      for (int i = 0; i < additionalInputCount; ++i) {
        auto noneNode = graph->createNone();
        noneNode->insertBefore(newNode);
        newNode->addInput(noneNode->output());
      }

      // Create a new output node and replace all uses of self with it
      newNode->output()->copyMetadata(node->output());
      node->replaceAllUsesWith(newNode);
      node->inputs()[0]->replaceAllUsesAfterNodeWith(
          newNode, newNode->output());
      node->destroy();
    }
  }
}
} // namespace

// Handles special case of binary inplace ops, where the first input node
// has a lower type precedence than the second input node. When the
// inplace node is converted to a regular op, this information is lost and
// the resulting type is based on type precedence, just like regular ops.
// To avoid this loss of information, we add a cast node before the input
// node with the higher data type precedence, so that both the input types
// are the same.
// An example scenario would be:
// Before:
// graph(%0 : Float),
//        %1 : Half):
//   %4 : Float = onnx::Cast[to=1](%1)
//   %5 : Float = onnx::Add(%4, %0)
//   ...
// After:
// graph(%0 : Float),
//        %1 : Half):
//   %4 : Half = onnx::Cast[to=10](%0)
//   %5 : Half = onnx::Add(%1, %4)
//   ...

void ImplicitCastForBinaryInplaceOps(Block* b) {
  for (auto it = b->nodes().begin(), end = b->nodes().end(); it != end; ++it) {
    for (auto* child_block : it->blocks()) {
      ImplicitCastForBinaryInplaceOps(child_block);
    }

    // Check type if inplace operation is a binary node
    if ((it->kind() == aten::add_) || (it->kind() == aten::sub_) ||
        (it->kind() == aten::mul_) || (it->kind() == aten::div_)) {
      auto orignalInputs = it->inputs();
      if (orignalInputs.at(0) == orignalInputs.at(1)) {
        continue;
      }
      TensorTypePtr firstInp_tensor =
          orignalInputs.at(0)->type()->cast<TensorType>();
      TensorTypePtr secondInp_tensor =
          orignalInputs.at(1)->type()->cast<TensorType>();
      if (!(firstInp_tensor) || !(secondInp_tensor) ||
          !(firstInp_tensor->scalarType().has_value())) {
        continue;
      }
      auto newInputNode = it->owningGraph()->create(aten::type_as, 1);
      newInputNode->insertBefore(*it);
      newInputNode->addInput(orignalInputs.at(1));
      newInputNode->addInput(orignalInputs.at(0));
      it->replaceInput(1, newInputNode->outputs().at(0));
    }
  }
}

void RemoveInplaceOps(const std::shared_ptr<Graph>& graph) {
  ImplicitCastForBinaryInplaceOps(graph->block());
  RemoveInplaceOps(graph->block());
}
} // namespace jit
} // namespace torch
