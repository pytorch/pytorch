#include "basic_ops.h"

#include "torch/csrc/autograd/variable.h"
#include "torch/csrc/autograd/functions/utils.h"
#include "torch/csrc/utils/auto_gpu.h"

namespace torch { namespace autograd {

auto Error::apply(const variable_list& grad_outputs) -> variable_list {
  throw std::runtime_error(msg);
};

auto DelayedError::apply(const variable_list& inputs) -> variable_list {
  tensor_list outputs;
  outputs.reserve(inputs.size());
  for (auto& var : inputs) {
    outputs.emplace_back(var ? var->data->clone_shallow() : nullptr);
  }
  return wrap_outputs(inputs, std::move(outputs), [&](FunctionFlags f) {
    return std::make_shared<Error>(msg, std::move(f));
  });
};

auto Add::apply(const variable_list& inputs) -> variable_list {
  check_input_variables("Add", inputs, 2);
  auto& input1 = inputs[0]->data;
  auto& input2 = inputs[1]->data;
  AutoGPU guard(input1->getDevice());

  bool first_sparse = input1->isSparse();
  auto output = first_sparse ? input2->newTensor() : input1->newTensor();
  if (first_sparse) {
    output->cadd(*input2, *input1);
  } else {
    output->cadd(*input1, *input2);
  }

  return wrap_outputs(inputs, as_tensor_list(std::move(output)), [&](FunctionFlags f) {
    return std::make_shared<AddBackward>(std::move(f));
  });
};

auto AddBackward::apply(const variable_list& grad_outputs) -> variable_list {
  check_input_variables("AddBackward", grad_outputs, 1);
  return {grad_outputs[0], grad_outputs[0]};
};

}} // namespace torch::autograd
