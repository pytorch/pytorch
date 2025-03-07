#pragma once

#include <torch/script.h>
#include "c10/core/Device.h"
#include "torch/csrc/runtime/executor/ExecutionFrame.h"
#include "torch/csrc/runtime/executor/OpKernel.h"

namespace torch::runtime {

class UnsafeAutoFunctionalizeKernel : public OpKernel {
 public:
  UnsafeAutoFunctionalizeKernel() = delete; // deleted default constructor
  UnsafeAutoFunctionalizeKernel(const Node* node);

  void computeInternal(ExecutionFrame& executionFrame) const override final;

 private:
  c10::OperatorHandle op_;
  c10::FunctionSchema schema_;

  Arguments arguments_;

  std::vector<Value*> mutatingInputArgs_;
  int numOutputs_;
};

} // namespace torch::runtime
