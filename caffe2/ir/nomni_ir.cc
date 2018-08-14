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

namespace nom {

void convert(torch::jit::Graph& jit_graph, NeuralNet* nn) {
  auto* block = jit_graph.block();

  auto basic_block =
      nn->controlFlow.createNode(repr::BasicBlockType<DFGraph>());
  auto nn_bb = basic_block->mutableData();

  std::unordered_map<torch::jit::Value*, nom::DFGraph::NodeRef> output_map;
  for (auto it = block->nodes().begin(); it != block->nodes().end(); ++it) {
    auto* node = *it;
    auto nn_op = nn->dataFlow.createNode(torch::jit::Node(node->kind()));
    nn_bb->pushInstructionNode(nn_op);

    for (auto* input : node->inputs()) {
      if (output_map.find(input) == output_map.end()) {
        auto nn_input = nn->dataFlow.createNode(torch::jit::Value());
        output_map[input] = nn_input;
      }
      nn->dataFlow.createEdge(output_map[input], nn_op);
    }

    for (auto* output : node->outputs()) {
      auto nn_val = nn->dataFlow.createNode(torch::jit::Value());
      output_map[output] = nn_val;
      nn->dataFlow.createEdge(nn_op, nn_val);
    }
  }
}

} // namespace nom
