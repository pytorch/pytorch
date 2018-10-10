#include <ATen/core/ir.h>
#include "caffe2/core/logging.h"

IRNode::IRNode(IRNode&& other) noexcept : kind_(other.kind_) {
  if (kind_ == IRNodeKind::Operator) {
    new (&op_) torch::jit::FunctionSchema(std::move(other.op_));
  } else if (kind_ == IRNodeKind::Value) {
    new (&value_) torch::jit::IValue(std::move(other.value_));
  }
};

IRNode::IRNode(torch::jit::FunctionSchema&& node)
    : kind_(IRNodeKind::Operator) {
  new (&op_) torch::jit::FunctionSchema(std::move(node));
}

IRNode::IRNode(torch::jit::IValue&& value) : kind_(IRNodeKind::Value) {
  new (&value_) torch::jit::IValue(std::move(value));
}

IRNode::IRNode(const torch::jit::IValue& value) : kind_(IRNodeKind::Value) {
  value_ = value;
}

IRNodeKind IRNode::getKind() const {
  return kind_;
}
const torch::jit::FunctionSchema& IRNode::getOperatorSchema() const {
  CAFFE_ENFORCE(kind_ == IRNodeKind::Operator);
  return op_;
}
const torch::jit::IValue& IRNode::getValue() const {
  CAFFE_ENFORCE(kind_ == IRNodeKind::Value);
  return value_;
}
torch::jit::IValue* IRNode::getMutableValue() {
  CAFFE_ENFORCE(kind_ == IRNodeKind::Value);
  return &value_;
}

IRNode::~IRNode() {
  switch (kind_) {
    case IRNodeKind::Operator:
      op_.~FunctionSchema();
      break;
    case IRNodeKind::Value:
      value_.~IValue();
      break;
      // Intentionally avoid default to raise compiler
      // error if new types are added to this wrapper.
  }
}

namespace nom {

const torch::jit::IValue& getInput(NeuralNet::NodeRef n, int index) {
  CAFFE_ENFORCE(n->data().getKind() == IRNodeKind::Operator);
  const auto& edges = n->getInEdges();
  CAFFE_ENFORCE(edges.size() > index);
  return getValue(edges[index]->tail());
}

std::vector<const torch::jit::IValue*> getInputs(NeuralNet::NodeRef n) {
  CAFFE_ENFORCE(n->data().getKind() == IRNodeKind::Operator);
  const auto& edges = n->getInEdges();
  std::vector<const torch::jit::IValue*> inputs;
  for (auto edge : edges) {
    inputs.emplace_back(&getValue(edge->tail()));
  }
  return inputs;
}

const torch::jit::IValue& getOutput(NeuralNet::NodeRef n, int index) {
  CAFFE_ENFORCE(n->data().getKind() == IRNodeKind::Operator);
  const auto& edges = n->getOutEdges();
  CAFFE_ENFORCE(edges.size() > index);
  return getValue(edges[index]->head());
}

torch::jit::IValue* getMutableOutput(NeuralNet::NodeRef n, int index) {
  CAFFE_ENFORCE(n->data().getKind() == IRNodeKind::Operator);
  const auto& edges = n->getOutEdges();
  CAFFE_ENFORCE(edges.size() > index);
  return getMutableValue(edges[index]->head());
}

std::vector<torch::jit::IValue*> getOutputs(NeuralNet::NodeRef n) {
  CAFFE_ENFORCE(n->data().getKind() == IRNodeKind::Operator);
  const auto& edges = n->getOutEdges();
  std::vector<torch::jit::IValue*> outputs;
  for (auto edge : edges) {
    outputs.emplace_back(getMutableValue(edge->head()));
  }
  return outputs;
}

const torch::jit::IValue& getValue(NeuralNet::NodeRef n) {
  return n->data().getValue();
}

torch::jit::IValue* getMutableValue(NeuralNet::NodeRef n) {
  return n->mutableData()->getMutableValue();
}

const torch::jit::FunctionSchema* getOperatorSchema(NeuralNet::NodeRef n) {
  return &n->data().getOperatorSchema();
}

} // namespace nom
