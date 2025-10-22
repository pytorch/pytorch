#pragma once

#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/core/function_schema.h>
#include <c10/core/Device.h>

#include <torch/nativert/executor/ExecutionFrame.h>
#include <torch/nativert/executor/OpKernel.h>

namespace torch::nativert {

class UnsafeAutoFunctionalizeKernel : public OpKernel {
 public:
  UnsafeAutoFunctionalizeKernel() = delete; // deleted default constructor
  UnsafeAutoFunctionalizeKernel(const Node* node);

  void computeInternal(ExecutionFrame& executionFrame) const override final;

 private:
  c10::OperatorHandle op_;
  c10::FunctionSchema schema_;

  Arguments arguments_;

  std::vector<Value*> mutatingInputArgs_;  // For v1: direct mutating input values
  int numOutputs_;
  bool isV2_ = false;  // Flag to track if this is auto_functionalized_v2
};

} // namespace torch::nativert
