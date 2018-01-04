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
    // FIXME: share version counters
    outputs.emplace_back(var.defined() ? var.data() : Tensor());
  }
  return wrap_outputs(inputs, std::move(outputs), [&](FunctionFlags f) {
    return std::make_shared<Error>(msg, std::move(f));
  });
};

}} // namespace torch::autograd
