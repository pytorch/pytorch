#include "torch/csrc/nativert/kernels/AOTICallDelegateKernel.h"

#include <c10/util/Logging.h>

#include "torch/csrc/nativert/executor/AOTIDelegateExecutor.h"
#include "torch/csrc/nativert/executor/ExecutionFrame.h"
#include "torch/csrc/nativert/executor/OpKernel.h"

namespace torch::nativert {

AOTICallDelegateKernel::AOTICallDelegateKernel(
    const Node* node,
    AOTIDelegateExecutor& delegateExecutor)
    : OpKernel(node), delegateExecutor_(delegateExecutor) {
  // torch.higher_order_ops.aoti_call_delegate(lowered_module, original_gm,
  // weight_args, input_args). However, the second
  // input is currently serialized as None, so we only have 3 inputs and 1
  // output.
  // TODO(T213681594): Fix this, None input should still be included in
  // numInputs().
  TORCH_CHECK_EQ(node->numInputs(), 3);
  TORCH_CHECK_EQ(node->numOutputs(), 1);

  // Weights are in node->inputs()[1], but it's not used in the forward call
  // Instead, weight are bound to AOTI via loadWeights()
  const Value* input = node->inputs()[2].value;
  const Value* output = node->outputs()[0];

  CHECK(input->type() == Type::TensorList)
      << "torch.higher_order_ops.aoti_call_delegate input should be a TensorList, but got "
      << input->type();
  CHECK(output->type() == Type::TensorList)
      << "torch.higher_order_ops.aoti_call_delegate output should be a TensorList, but got "
      << output->type();

  inputValueId_ = input->id();
  outputValueId_ = output->id();
}

void AOTICallDelegateKernel::computeInternal(
    ExecutionFrame& executionFrame) const {
  std::vector<at::Tensor> inputs =
      executionFrame.getTensorVector(inputValueId_);

  auto outputs = delegateExecutor_.run(inputs);

  executionFrame.setIValue(outputValueId_, std::move(outputs));
}

} // namespace torch::nativert
