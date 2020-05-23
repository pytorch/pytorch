#pragma once

#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/onnx/onnx.h>
#include <vector>

namespace torch { namespace autograd {

struct SymbolicContext {
  jit::Block* block;
};

struct symbolic_unconvertible : public std::runtime_error {
  using std::runtime_error::runtime_error;
};

}} // namespace torch::autograd
