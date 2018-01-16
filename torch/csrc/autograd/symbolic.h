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

}} // namespace torch::autograd
