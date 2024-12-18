#include <torch/csrc/lazy/core/ops/arithmetic_ir_ops.h>

#include <torch/csrc/lazy/core/helpers.h>

#include <memory>

#include <torch/csrc/lazy/core/ir_builder.h>

namespace torch::lazy {

// These operators were once widely used in nativefunction impls to perform
// convenient decompositions (partial lowerings) of aten operators into more
// primitive opererators. They should not be used for this purpose anymore, but
// still used in lazy_graph_executor for RNG math in one place.  We could
// rewrite that.
NodePtr operator+(const Value& node1, const Value& node2) {
  return MakeGeneric(
      OpKind(at::aten::add),
      {node1, node2},
      GetPromotedBinaryOpShape(node1.shape(), node2.shape()));
}

NodePtr operator-(const Value& node1, const Value& node2) {
  return MakeGeneric(
      OpKind(at::aten::sub),
      {node1, node2},
      GetPromotedBinaryOpShape(node1.shape(), node2.shape()));
}

NodePtr operator*(const Value& node1, const Value& node2) {
  return MakeGeneric(
      OpKind(at::aten::mul),
      {node1, node2},
      GetPromotedBinaryOpShape(node1.shape(), node2.shape()));
}

NodePtr operator/(const Value& node1, const Value& node2) {
  return MakeGeneric(
      OpKind(at::aten::div),
      {node1, node2},
      GetPromotedBinaryOpShape(node1.shape(), node2.shape()));
}

} // namespace torch::lazy
