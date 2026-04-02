#pragma once

#include <torch/nativert/executor/ExecutionFrame.h>
#include <torch/nativert/executor/OpKernel.h>

namespace torch::nativert {

class ETDelegateExecutor;

class ETCallDelegateKernel : public OpKernel {
 public:
  explicit ETCallDelegateKernel(
      const Node* node,
      ETDelegateExecutor& delegateExecutor);

  void computeInternal(ExecutionFrame& executionFrame) const final;

 private:
  ETDelegateExecutor& delegateExecutor_;
};

} // namespace torch::nativert
