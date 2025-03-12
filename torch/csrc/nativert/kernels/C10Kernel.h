

#pragma once

#include <torch/script.h>
#include "c10/core/Device.h"
#include "torch/csrc/nativert/executor/ExecutionFrame.h"
#include "torch/csrc/nativert/executor/OpKernel.h"
#include "torch/csrc/nativert/graph/Graph.h"

namespace torch::nativert {

// Implementation of Kernel for ATen operators
//
// This class exists to amortize per-kernel overhead by computing things during
// initialization instead of on every execution. Right now we are only
// amortizing schema resolution, and static arguments parsing,
// but in the future this could be extended to avoid operator dispatch and
// do better "Register" allocation (e.g. convert input/outputs to directly
// array accesses onto a set of registers, in concert with memory planning)
class C10Kernel : public OpKernel {
 public:
  C10Kernel() = delete; // deleted default constructor
  C10Kernel(const Node* node, c10::Device device);
  virtual ~C10Kernel() = default;

  [[nodiscard]] const c10::IValue& input(
      uint32_t i,
      ExecutionFrame& executionFrame) const override {
    if (Value* dynamicArg = arguments_.findDynamic(i)) {
      return executionFrame.getIValue(dynamicArg->id());
    } else {
      return arguments_.getStatic(i);
    }
  }

  void computeInternal(ExecutionFrame& executionFrame) const override;

 private:
  c10::OperatorHandle op_;
  c10::FunctionSchema schema_;

  Arguments arguments_;
};

class SymIntOpKernel : public OpKernel {
 public:
  explicit SymIntOpKernel(const Node* node) : OpKernel(node) {}
  void computeInternal(ExecutionFrame& executionFrame) const override final;
};

class SymBoolOpKernel : public OpKernel {
 public:
  explicit SymBoolOpKernel(const Node* node) : OpKernel(node) {}
  void computeInternal(ExecutionFrame& executionFrame) const override final;
};

class SymFloatOpKernel : public OpKernel {
 public:
  explicit SymFloatOpKernel(const Node* node) : OpKernel(node) {}
  void computeInternal(ExecutionFrame& executionFrame) const override final;
};

// ScalarOpKernel does binary arithmetic operations on scalar values.
// Integers and floats are supported as input types. The output will be
// promoted to float if and only if there's at least one float input.
class ScalarBinaryOpKernel : public OpKernel {
 public:
  explicit ScalarBinaryOpKernel(const Node* node) : OpKernel(node) {}
  void computeInternal(ExecutionFrame& executionFrame) const override final;
};

} // namespace torch::nativert
