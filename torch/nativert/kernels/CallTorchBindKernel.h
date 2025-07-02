#pragma once

#include <c10/core/Device.h>
#include <torch/custom_class.h>

#include <torch/nativert/executor/ExecutionFrame.h>
#include <torch/nativert/executor/OpKernel.h>

namespace torch::nativert {

class CallTorchBindKernel : public OpKernel {
 public:
  CallTorchBindKernel() = delete; // deleted default constructor
  CallTorchBindKernel(const Node* node);

  void computeInternal(ExecutionFrame& executionFrame) const override final;

 private:
  std::string methodName_;
  torch::jit::Function* method_;

  std::string customClassName_;
  at::ClassTypePtr customClassType_;
};

} // namespace torch::nativert
