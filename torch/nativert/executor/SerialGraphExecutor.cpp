#include <ATen/record_function.h>
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
  executionFrame.withManagedMemory([&](const LayoutManager* layout_manager) {
    // Execute kernels for all nodes except prim.Input and prim.Output
    for (NodeIndex nodeIdx = 1; nodeIdx < nodeKernels_.size() - 1; ++nodeIdx) {
      if (executorConfig_.enableOpProfiling) {
        RECORD_FUNCTION(
            nodeKernels_[nodeIdx]->node()->target(),
            c10::ArrayRef<const c10::IValue>{});
      }
      nodeKernels_[nodeIdx]->compute(executionFrame);

#ifndef NDEBUG
      if (layout_manager != nullptr) {
        layout_manager->assert_no_overlapping_storages(nodeIdx);
      }
#endif

      // don't free intermediate values when static memory planning is enabled
      if (executorConfig_.tryFreeUnmanagedValuesAfterUse) {
        // Free the intermediate values that are no used anymore
        for (const auto& valueKey : execPlan_->valuesToFree[nodeIdx]) {
          executionFrame.releaseValueIfNeeded(valueKey);
        }
      }
    }
  });
  return executionFrame.tryMoveUserOutputs();
}

} // namespace torch::nativert
