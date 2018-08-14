#include "caffe2/ir/nomni_ir.h"

IRNode::IRNode(IRNode&& other) : kind_(other.kind_) {
  if (kind_ == IRNodeKind::Operator) {
    new (&op_) torch::jit::Node(std::move(other.op_));
  } else if (kind_ == IRNodeKind::Value) {
    new (&value_) torch::jit::Value(std::move(other.value_));
  }
};

IRNode::IRNode(torch::jit::Node&& node) : kind_(IRNodeKind::Operator) {
  new (&op_) torch::jit::Node(std::move(node));
}

IRNode::IRNode(torch::jit::Value&& value) : kind_(IRNodeKind::Value) {
  new (&value_) torch::jit::Value(std::move(value));
}

IRNodeKind IRNode::getKind() const {
  return kind_;
}
const torch::jit::Node& IRNode::getOperator() const {
  CAFFE_ENFORCE(kind_ == IRNodeKind::Operator);
  return op_;
}
const torch::jit::Value& IRNode::getValue() const {
  CAFFE_ENFORCE(kind_ == IRNodeKind::Value);
  return value_;
}

IRNode::~IRNode() {
  switch (kind_) {
    case IRNodeKind::Operator:
      op_.~Node();
      break;
    case IRNodeKind::Value:
      value_.~Value();
      break;
      // Intentionally avoid default to raise compiler
      // error if new types are added to this wrapper.
  }
}
