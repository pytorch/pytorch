#include "torch/csrc/nativert/kernels/AOTIKernel.h"

#include <c10/util/Logging.h>

#include "torch/csrc/nativert/executor/AOTIDelegateExecutor.h"
#include "torch/csrc/nativert/executor/ExecutionFrame.h"
#include "torch/csrc/nativert/executor/OpKernel.h"

namespace torch::nativert {

AOTIKernel::AOTIKernel(const Node* node, AOTIDelegateExecutor& delegateExecutor)
    : OpKernel(node), delegateExecutor_(delegateExecutor) {
  // The schema is "call_aotinductor(str path, Tensor[] weights, Tensor[]
  // inputs) -> Tensor[] outputs", expects 2 inputs and 1 output
  TORCH_CHECK_EQ(node->numInputs(), 2);
  TORCH_CHECK_EQ(node->numOutputs(), 1);

  // Weights are in node->inputs()[0], but it's not used in the forward call
  // Instead, weight are bound to AOTI via loadWeights()
  const Value* input = node->inputs()[1].value;
  const Value* output = node->outputs()[0];

  CHECK(input->type() == Type::TensorList)
      << "delegate.call_aotinductor input should be a TensorList";
  CHECK(output->type() == Type::TensorList)
      << "delegate.call_aotinductor output should be a TensorList";

  inputValueId_ = input->id();
  outputValueId_ = output->id();
}

void AOTIKernel::computeInternal(ExecutionFrame& executionFrame) const {
  std::vector<at::Tensor> inputs =
      executionFrame.getTensorVector(inputValueId_);

  auto outputs = delegateExecutor_.run(inputs);

  executionFrame.setIValue(outputValueId_, std::move(outputs));
}

} // namespace torch::nativert
