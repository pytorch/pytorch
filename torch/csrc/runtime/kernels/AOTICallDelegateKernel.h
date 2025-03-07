#pragma once

#include "torch/csrc/runtime/executor/ExecutionFrame.h"

#include "torch/csrc/runtime/executor/OpKernel.h"

namespace torch::runtime {

class AOTIDelegateExecutor;

// Kernel for torch.higher_order_ops.aoti_call_delegate
class AOTICallDelegateKernel : public OpKernel {
 public:
  explicit AOTICallDelegateKernel(
      const Node* node,
      AOTIDelegateExecutor& delegateExecutor);

  void computeInternal(ExecutionFrame& executionFrame) const override final;

 private:
  AOTIDelegateExecutor& delegateExecutor_;

  ValueId inputValueId_;
  ValueId outputValueId_;
};

} // namespace torch::runtime
