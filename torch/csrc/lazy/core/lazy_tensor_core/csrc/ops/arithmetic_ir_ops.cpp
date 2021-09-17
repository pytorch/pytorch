#include "lazy_tensor_core/csrc/ops/arithmetic_ir_ops.h"

#include <memory>

#include "lazy_tensor_core/csrc/helpers.h"
#include "lazy_tensor_core/csrc/ops/ops.h"

namespace torch_lazy_tensors {
namespace ir {

NodePtr operator+(const Value& node1, const Value& node2) {
  return ops::GenericOp(
      OpKind(at::aten::add), {node1, node2},
      Helpers::GetPromotedBinaryOpShape(node1.shape(), node2.shape()));
}

NodePtr operator-(const Value& node1, const Value& node2) {
  return ops::GenericOp(
      OpKind(at::aten::sub), {node1, node2},
      Helpers::GetPromotedBinaryOpShape(node1.shape(), node2.shape()));
}

NodePtr operator*(const Value& node1, const Value& node2) {
  return ops::GenericOp(
      OpKind(at::aten::mul), {node1, node2},
      Helpers::GetPromotedBinaryOpShape(node1.shape(), node2.shape()));
}

NodePtr operator/(const Value& node1, const Value& node2) {
  return ops::GenericOp(
      OpKind(at::aten::div), {node1, node2},
      Helpers::GetPromotedBinaryOpShape(node1.shape(), node2.shape()));
}

}  // namespace ir
}  // namespace torch_lazy_tensors
