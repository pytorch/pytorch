#include <torch/nativert/kernels/AutoFunctionalizeKernel.h>

#include <fmt/format.h>

#include <c10/util/Enumerate.h>

namespace torch::nativert {

UnsafeAutoFunctionalizeKernel::UnsafeAutoFunctionalizeKernel(const Node* node)
    : OpKernel(node),
      op_(getOperatorForTarget(
          std::get<std::string>(node->attributes()[0].value))),
      schema_(op_.schema()),
      arguments_(prefillStackWithStaticArgs(node, schema_)),
      numOutputs_(static_cast<int>(schema_.returns().size())) {
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
    throw std::runtime_error(fmt::format(
        "Original Python stacktrace:\n{}\n{}",
        stackTrace ? *stackTrace : "<no stack trace>",
        ex.what()));
  }

  const auto& outputValues = node_->outputs();

  for (int i = 0; i < numOutputs_; ++i) {
    executionFrame.setIValue(outputValues[i]->id(), std::move(stack.at(i)));
  }

  // Copy over mutating inputs to outputs
  int mutatingArgStartIndex = (numOutputs_ == 0) ? 1 : numOutputs_;
  for (size_t i = mutatingArgStartIndex; i < outputValues.size(); ++i) {
    executionFrame.setIValue(
        outputValues[i]->id(),
        executionFrame.getIValue(
            mutatingInputArgs_.at(i - mutatingArgStartIndex)->id(),
            true /*  allowNone */));
  }
}

} // namespace torch::nativert
