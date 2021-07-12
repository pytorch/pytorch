#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/remove_inplace_ops.h>

namespace torch {
namespace jit {
// Handles special case of binary inplace ops, where the first input node
// has a lower type precedence than the second input node. When the
// inplace node is converted to a regular op, this information is lost and
// the resulting type is based on type precedence, just like regular ops.
// To avoid this loss of information, we add a cast node before the input
// node with the higher data type precedence, so that both the input types
// are the same.
// An example scenario would be:
// Before:
// graph(%0 : Half),
//       %1 : Float):
//   # Should result in a Half, but after translation to out-of-place,
//   # would become a Float b/c Half+Float -> Float.
//   Float : = aten::add_(%0, %1)
//   ...
// After:
// graph(%0 : Half),
//       %1 : Float):
//   %2 : Half = aten::type_as(%1, %0)
//   # Half + Half will result in correct dtype.
//   Half : = aten::add_(%0, %2)
//   ...
void ImplicitCastForBinaryInplaceOps(Block* b) {
  for (auto it = b->nodes().begin(), end = b->nodes().end(); it != end; ++it) {
    for (auto* child_block : it->blocks()) {
      ImplicitCastForBinaryInplaceOps(child_block);
    }

    // Check type if inplace operation is a binary node
    if ((it->kind() == aten::add_) || (it->kind() == aten::sub_) ||
        (it->kind() == aten::mul_) || (it->kind() == aten::div_)) {
      auto originalInputs = it->inputs();
      if (originalInputs.at(0) == originalInputs.at(1)) {
        continue;
      }
      TensorTypePtr firstInp_tensor =
          originalInputs.at(0)->type()->cast<TensorType>();
      TensorTypePtr secondInp_tensor =
          originalInputs.at(1)->type()->cast<TensorType>();
      if (!(firstInp_tensor) || !(secondInp_tensor) ||
          !(firstInp_tensor->scalarType().has_value())) {
        continue;
      }
      auto newInputNode = it->owningGraph()->create(aten::type_as, 1);
      newInputNode->insertBefore(*it);
      newInputNode->addInput(originalInputs.at(1));
      newInputNode->addInput(originalInputs.at(0));
      it->replaceInput(1, newInputNode->outputs().at(0));
    }
  }
}
} // namespace jit
} // namespace torch
