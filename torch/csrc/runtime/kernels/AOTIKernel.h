#pragma once

#include "torch/csrc/runtime/executor/ExecutionFrame.h"

#include "torch/csrc/runtime/executor/OpKernel.h"

namespace torch::runtime {

class AOTIDelegateExecutor;

// Kernel for torch.ops.delegate.call_aotinductor
// TODO: Deprecate this when we move to aoti_call_delegate HOP
class AOTIKernel : public OpKernel {
 public:
  explicit AOTIKernel(const Node* node, AOTIDelegateExecutor& delegateExecutor);

  void computeInternal(ExecutionFrame& executionFrame) const override final;

 private:
  AOTIDelegateExecutor& delegateExecutor_;

  ValueId inputValueId_;
  ValueId outputValueId_;
};

} // namespace torch::runtime
