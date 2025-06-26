#include <torch/nativert/executor/ExecutionPlanner.h>
#include <torch/nativert/executor/ExecutorConfig.h>
#include <torch/nativert/executor/SerialGraphExecutor.h>

namespace torch::nativert {

std::vector<c10::IValue> SerialGraphExecutor::execute(
    ExecutionFrame& executionFrame,
    std::vector<c10::IValue> inputs) {
  fillUserInputs(executionFrame, std::move(inputs));

  return executeWithPrefilledFrame(executionFrame);
}

std::vector<c10::IValue> SerialGraphExecutor::executeWithPrefilledFrame(
    ExecutionFrame& executionFrame) {
  // Execute kernels for all nodes except prim.Input and prim.Output
  for (NodeIndex nodeIdx = 1; nodeIdx < nodeKernels_.size() - 1; ++nodeIdx) {
    nodeKernels_[nodeIdx]->compute(executionFrame);

    // don't free intermediate values when static memory planning is enabled
    if (executorConfig_.tryFreeUnmanagedValuesAfterUse) {
      // Free the intermediate values that are no used anymore
      for (const auto& valueKey : execPlan_->valuesToFree[nodeIdx]) {
        executionFrame.releaseValue(valueKey);
      }
    }
  }

  return executionFrame.tryMoveUserOutputs();
}

} // namespace torch::nativert
