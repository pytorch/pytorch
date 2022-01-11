#include <torch/csrc/lazy/ts_backend/ops/arithmetic_ir_ops.h>

#include <torch/csrc/lazy/core/helpers.h>

#include <memory>

#include <torch/csrc/lazy/ts_backend/ops/generic.h>

namespace torch {
namespace lazy {

NodePtr operator+(const Value& node1, const Value& node2) {
  return GenericOp(
      OpKind(at::aten::add),
      {node1, node2},
      GetPromotedBinaryOpShape(
          GetShapeFromTsValue(node1), GetShapeFromTsValue(node2)));
}

NodePtr operator-(const Value& node1, const Value& node2) {
  return GenericOp(
      OpKind(at::aten::sub),
      {node1, node2},
      GetPromotedBinaryOpShape(
          GetShapeFromTsValue(node1), GetShapeFromTsValue(node2)));
}

NodePtr operator*(const Value& node1, const Value& node2) {
  return GenericOp(
      OpKind(at::aten::mul),
      {node1, node2},
      GetPromotedBinaryOpShape(
          GetShapeFromTsValue(node1), GetShapeFromTsValue(node2)));
}

NodePtr operator/(const Value& node1, const Value& node2) {
  return GenericOp(
      OpKind(at::aten::div),
      {node1, node2},
      GetPromotedBinaryOpShape(
          GetShapeFromTsValue(node1), GetShapeFromTsValue(node2)));
}

} // namespace lazy
} // namespace torch
