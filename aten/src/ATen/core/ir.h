#pragma once

#include <ATen/core/function_schema.h>
#include <ATen/core/ivalue.h>
#include "nomnigraph/Graph/Graph.h"
#include "nomnigraph/Representations/ControlFlow.h"

enum class IRNodeKind { Operator, Value };

// Purely moveable tagged union to wrap torch::jit::{FunctionSchema,IValue}
struct CAFFE2_API IRNode {
  IRNode() = delete;
  IRNode(const IRNode&) = delete;
  IRNode& operator=(const IRNode& other) = delete;
  IRNode(IRNode&& other) noexcept;
  IRNode& operator=(IRNode&& other) = default;

  IRNode(torch::jit::FunctionSchema&& node);
  IRNode(torch::jit::IValue&& value);
  IRNode(const torch::jit::IValue& value);

  // If you aren't sure what is stored at a node,
  // query this method first.
  IRNodeKind getKind() const;

  // These methods raise exceptions if miscalled.
  const torch::jit::FunctionSchema& getOperatorSchema() const;
  const torch::jit::IValue& getValue() const;
  // TODO: const correctness here should be discussed thoroughly
  torch::jit::IValue* getMutableValue();

  // Nontrivial destructor.
  ~IRNode();

 private:
  const IRNodeKind kind_;
  union {
    torch::jit::FunctionSchema op_;
    torch::jit::IValue value_;
  };
};

namespace nom {

using DFGraph = nom::Graph<IRNode>;
using CFGraph = nom::repr::ControlFlowGraph<DFGraph>;
struct NeuralNet {
  using NodeRef = DFGraph::NodeRef;
  DFGraph dataFlow;
  CFGraph controlFlow;
};

const torch::jit::IValue& getValue(NeuralNet::NodeRef n);
torch::jit::IValue* getMutableValue(NeuralNet::NodeRef n);
const torch::jit::IValue& getInput(NeuralNet::NodeRef n, int index);
std::vector<const torch::jit::IValue*> getInputs(NeuralNet::NodeRef n);
const torch::jit::IValue& getOutput(NeuralNet::NodeRef n, int index);
torch::jit::IValue* getMutableOutput(NeuralNet::NodeRef n, int index);
std::vector<torch::jit::IValue*> getOutputs(NeuralNet::NodeRef n);
const torch::jit::FunctionSchema* getOperatorSchema(NeuralNet::NodeRef n);

} // namespace nom
