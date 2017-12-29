#pragma once

#include "torch/csrc/jit/ir.h"
#include "torch/csrc/onnx/onnx.h"
#include <vector>

namespace torch { namespace autograd {

struct SymbolicContext {
  jit::Graph* graph;
};

struct symbolic_unconvertible : public std::runtime_error {
  using std::runtime_error::runtime_error;
};


struct HasSymbolic {
  // Add some nodes to the ONNX protobuf, under the assumption that this node
  // as a whole has the represented inputs and outputs.  Raises a
  // symbolic_unconvertible exception if conversion is not supported.
  virtual jit::value_list symbolic(
      SymbolicContext* ctx,
      jit::value_list inputs,
      std::shared_ptr<jit::SourceLocation> sl
  ) = 0;
};

}} // namespace torch::autograd
