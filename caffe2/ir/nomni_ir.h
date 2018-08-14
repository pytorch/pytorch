#pragma once

#include "caffe2/core/logging.h"
#include "nomnigraph/Graph/Graph.h"
#include "nomnigraph/Representations/ControlFlow.h"
#include "torch/csrc/jit/ir.h"

enum class IRNodeKind { Operator, Value };

// Purely moveable tagged union to wrap torch::jit::{Node,Value}
struct IRNode {
  IRNode() = delete;
  IRNode(const IRNode&) = delete;
  IRNode(IRNode&& other);

  IRNode(torch::jit::Node&& node);
  IRNode(torch::jit::Value&& value);

  // If you aren't sure what is stored at a node,
  // query this method first.
  IRNodeKind getKind() const;

  // These methods raise exceptions if miscalled.
  const torch::jit::Node& getOperator() const;
  const torch::jit::Value& getValue() const;

  // Nontrivial destructor.
  ~IRNode();

 private:
  const IRNodeKind kind_;
  union {
    torch::jit::Node op_;
    torch::jit::Value value_;
  };
};

namespace nom {

using DFGraph = nom::Graph<IRNode>;
using CFGraph = nom::repr::ControlFlowGraph<DFGraph>;
struct NeuralNet {
  DFGraph dataFlow;
  CFGraph controlFlow;
};

} // namespace nom
