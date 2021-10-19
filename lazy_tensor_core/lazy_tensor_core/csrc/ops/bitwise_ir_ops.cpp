#include "lazy_tensor_core/csrc/ops/bitwise_ir_ops.h"

#include "lazy_tensor_core/csrc/compiler/node_lowering.h"
#include "lazy_tensor_core/csrc/helpers.h"
#include "lazy_tensor_core/csrc/ops/ops.h"
#include "lazy_tensor_core/csrc/ts_backend/TsNode.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

torch::lazy::Value BitwiseOp(const OpKind& kind, const torch::lazy::Value& node1, const torch::lazy::Value& node2) {
  NodePtr node = GenericOp(kind, {node1, node2});
  ir::TsNodeSetShapeDeferred(
      node, [&]() { return compiler::NodeLowering::Get()->Infer(node.get()); });
  return node;
}

torch::lazy::Value BitwiseOr(const torch::lazy::Value& node1, const torch::lazy::Value& node2) {
  return BitwiseOp(OpKind(at::aten::__or__), node1, node2);
}

torch::lazy::Value BitwiseXor(const torch::lazy::Value& node1, const torch::lazy::Value& node2) {
  return BitwiseOp(OpKind(at::aten::__xor__), node1, node2);
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
