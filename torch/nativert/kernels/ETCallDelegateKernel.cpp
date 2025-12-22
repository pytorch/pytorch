#include <torch/nativert/kernels/ETCallDelegateKernel.h>

#include <torch/nativert/executor/ETDelegateExecutor.h>

namespace torch::nativert {

ETCallDelegateKernel::ETCallDelegateKernel(
    const Node* node,
    ETDelegateExecutor& delegateExecutor)
    : OpKernel(node), delegateExecutor_(delegateExecutor) {
  for (const auto& input : node_->inputs()) {
    TORCH_CHECK(input.value->type() == Type::Kind::Tensor);
  }

  for (const auto* output : node_->outputs()) {
    TORCH_CHECK(output->type() == Type::Kind::Tensor);
  }
}

void ETCallDelegateKernel::computeInternal(
    ExecutionFrame& executionFrame) const {
  std::vector<at::Tensor> inputs;
  inputs.reserve(numInputs());

  for (const auto& input : node_->inputs()) {
    inputs.emplace_back(executionFrame.getTensor(input.value->id()));
  }

  auto outputs = delegateExecutor_.run(inputs);
  const auto& node_outputs = node_->outputs();
  TORCH_CHECK(outputs.size() == node_outputs.size());

  size_t i = 0;
  for (auto begin = std::make_move_iterator(outputs.begin()),
            end = std::make_move_iterator(outputs.end());
       begin != end;
       ++begin) {
    executionFrame.setIValue(node_outputs[i]->id(), *begin);
    i++;
  }
}

} // namespace torch::nativert
