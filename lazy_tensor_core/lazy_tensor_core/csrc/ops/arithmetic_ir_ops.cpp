#include "lazy_tensor_core/csrc/ops/arithmetic_ir_ops.h"

#include <memory>

#include "lazy_tensor_core/csrc/helpers.h"
#include "lazy_tensor_core/csrc/ops/ops.h"

namespace torch_lazy_tensors {
namespace ir {

torch::lazy::NodePtr operator+(const torch::lazy::Value& node1, const torch::lazy::Value& node2) {
  return ops::GenericOp(
      OpKind(at::aten::add), {node1, node2},
      Helpers::GetPromotedBinaryOpShape(ir::GetShapeFromTsValue(node1), GetShapeFromTsValue(node2)));
}

torch::lazy::NodePtr operator-(const torch::lazy::Value& node1, const torch::lazy::Value& node2) {
  return ops::GenericOp(
      OpKind(at::aten::sub), {node1, node2},
      Helpers::GetPromotedBinaryOpShape(ir::GetShapeFromTsValue(node1), GetShapeFromTsValue(node2)));
}

torch::lazy::NodePtr operator*(const torch::lazy::Value& node1, const torch::lazy::Value& node2) {
  return ops::GenericOp(
      OpKind(at::aten::mul), {node1, node2},
      Helpers::GetPromotedBinaryOpShape(ir::GetShapeFromTsValue(node1), GetShapeFromTsValue(node2)));
}

torch::lazy::NodePtr operator/(const torch::lazy::Value& node1, const torch::lazy::Value& node2) {
  return ops::GenericOp(
      OpKind(at::aten::div), {node1, node2},
      Helpers::GetPromotedBinaryOpShape(ir::GetShapeFromTsValue(node1), GetShapeFromTsValue(node2)));
}

}  // namespace ir
}  // namespace torch_lazy_tensors
