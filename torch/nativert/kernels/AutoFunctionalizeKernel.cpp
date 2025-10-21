#include <torch/nativert/kernels/AutoFunctionalizeKernel.h>

#include <c10/util/Enumerate.h>
#include <c10/util/Exception.h>

namespace torch::nativert {

UnsafeAutoFunctionalizeKernel::UnsafeAutoFunctionalizeKernel(const Node* node)
    : OpKernel(node),
      op_(getOperatorForTarget(
          std::get<std::string>(node->attributes()[0].value))),
      schema_(op_.schema()),
      arguments_(prefillStackWithStaticArgs(node, schema_)),
      numOutputs_(static_cast<int>(schema_.returns().size())) {
  // Check if this is auto_functionalized_v2
  isV2_ = node->target().find("auto_functionalized_v2") != std::string::npos;

  for (const auto& [idx, schemaArg] : c10::enumerate(schema_.arguments())) {
    if (schemaArg.alias_info() != nullptr &&
        schemaArg.alias_info()->isWrite()) {
      mutatingInputArgs_.push_back(node->getInput(schemaArg.name()).value);
    }
  }
}

void UnsafeAutoFunctionalizeKernel::computeInternal(
    ExecutionFrame& executionFrame) const {
  // Make a copy of the stack
  std::vector<c10::IValue> stack = arguments_.getStackWithStaticArgs();

  fillDynamicInputs(executionFrame, arguments_, stack);

  // Call the op with the prepared stack.
  try {
    op_.callBoxed(stack);
  } catch (const std::exception& ex) {
    // TODO: this eats the original exception type. ATen returns different
    // exception types that correspond to different Python errors (e.g.
    // IndexError, ValueError). If retaining this information is important
    // to us, we'll have to change this up a little.
    auto stackTrace = node_->getMetadata("stack_trace");
    TORCH_CHECK(
        false,
        "Oringinal Python stacktrace:\n",
        stackTrace ? *stackTrace : "<no stack trace>",
        "\n",
        ex.what())
  }

  const auto& outputValues = node_->outputs();

  // For auto_functionalized_v2, the output structure is different
  // v1: returns (actual_outputs, *mutated_inputs)
  // v2: returns (actual_outputs, *all_bases) where all_bases are the mutated base tensors
  if (isV2_) {
    // For v2, the first numOutputs_ are the actual outputs from the operation
    // The remaining outputs are the mutated base tensors that need to be
    // mapped back to the corresponding output values in the graph

    // First, set the actual operation outputs
    for (int i = 0; i < numOutputs_; ++i) {
      executionFrame.setIValue(outputValues[i]->id(), std::move(stack.at(i)));
    }

    // Then handle the mutated base tensors
    // In v2, after the regular outputs come all the base tensors that were mutated
    // The graph's output values should get these base tensors
    int baseStartIndex = numOutputs_;
    int numBases = static_cast<int>(stack.size()) - numOutputs_;

    // Map the returned base tensors to the output values
    // Note: The graph's outputValues after numOutputs_ correspond to the mutated arguments
    // and should receive the corresponding base tensors from the stack
    for (int i = 0; i < numBases; ++i) {
      int outputIdx = numOutputs_ + i;
      if (outputIdx < static_cast<int>(outputValues.size())) {
        executionFrame.setIValue(
            outputValues[outputIdx]->id(),
            std::move(stack.at(baseStartIndex + i)));
      }
    }
  } else {
    // Original auto_functionalized (v1) behavior
    // v1 returns (actual_outputs, *mutated_inputs)

    // First set the actual outputs
    for (int i = 0; i < numOutputs_; ++i) {
      executionFrame.setIValue(outputValues[i]->id(), std::move(stack.at(i)));
    }

    // Then set the mutated inputs
    // v1 returns the mutated inputs directly after the outputs
    int mutatedStartIndex = numOutputs_;
    for (size_t i = 0; i < mutatingInputArgs_.size(); ++i) {
      int outputIdx = numOutputs_ + i;
      if (outputIdx < static_cast<int>(outputValues.size())) {
        executionFrame.setIValue(
            outputValues[outputIdx]->id(),
            std::move(stack.at(mutatedStartIndex + i)));
      }
    }
  }
}

} // namespace torch::nativert
