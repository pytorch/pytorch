#include "basic_ops.h"

#include "torch/csrc/autograd/variable.h"
#include "torch/csrc/autograd/functions/utils.h"
#include "torch/csrc/utils/auto_gpu.h"

namespace torch { namespace autograd {

auto Add::apply(const variable_list& inputs) -> variable_list {
  if (inputs.size() != 2) throw std::runtime_error("Add expects exactly 2 inputs");
  auto& input1 = inputs[0]->data;
  auto& input2 = inputs[1]->data;
  AutoGPU(input1->getDevice());

  auto output = input1->newTensor();
  output->cadd(*input1, *input2);


  return wrap_outputs(inputs, as_tensor_list(std::move(output)), [&](FunctionFlags f) {
    return std::make_shared<AddBackward>(f);
  });
};

auto AddBackward::apply(const variable_list& grad_outputs) -> variable_list {
  if (grad_outputs.size() != 1) throw std::runtime_error("AddBackward expects exactly 1 grad_output");
  return {grad_outputs[0], grad_outputs[0]};
};

}} // namespace torch::autograd

