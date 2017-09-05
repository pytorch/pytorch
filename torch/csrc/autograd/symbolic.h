#pragma once

#include <Python.h>
#include "torch/csrc/jit/ir.h"
#include "torch/csrc/onnx.h"
#include <vector>

namespace torch { namespace autograd {

struct SymbolicContext {
  jit::Graph* graph;
  const std::unordered_map<void*, jit::Node*>* buffer_map;
  int batch_norm_count = 0;
};

struct symbolic_unconvertible : public std::runtime_error {
  using std::runtime_error::runtime_error;
};


struct HasSymbolic {
  // Add some nodes to the ONNX protobuf, under the assumption that this node
  // as a whole has the represented inputs and outputs.  Raises a
  // symbolic_unconvertible exception if conversion is not supported.
  virtual jit::node_list symbolic(SymbolicContext* ctx, jit::node_list inputs) = 0;
};

}} // namespace torch::autograd
