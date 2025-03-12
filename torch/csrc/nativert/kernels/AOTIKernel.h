#pragma once

#include "torch/csrc/nativert/executor/ExecutionFrame.h"

#include "torch/csrc/nativert/executor/OpKernel.h"

namespace torch::nativert {

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

} // namespace torch::nativert
