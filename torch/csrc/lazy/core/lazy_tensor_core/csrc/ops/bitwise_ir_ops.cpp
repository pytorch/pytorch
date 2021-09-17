#include "lazy_tensor_core/csrc/ops/bitwise_ir_ops.h"

#include "lazy_tensor_core/csrc/compiler/node_lowering.h"
#include "lazy_tensor_core/csrc/helpers.h"
#include "lazy_tensor_core/csrc/ops/ops.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

Value BitwiseOp(const OpKind& kind, const Value& node1, const Value& node2) {
  NodePtr node = GenericOp(kind, {node1, node2});
  node->SetShapeDeferred(
      [&]() { return compiler::NodeLowering::Get()->Infer(node.get()); });
  return node;
}

Value BitwiseAnd(const Value& node1, const Value& node2) {
  return BitwiseOp(OpKind(at::aten::__and__), node1, node2);
}

Value BitwiseOr(const Value& node1, const Value& node2) {
  return BitwiseOp(OpKind(at::aten::__or__), node1, node2);
}

Value BitwiseXor(const Value& node1, const Value& node2) {
  return BitwiseOp(OpKind(at::aten::__xor__), node1, node2);
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
