#include "basic_ops.h"

#include "torch/csrc/autograd/function.h"
#include "torch/csrc/autograd/variable.h"
#include "torch/csrc/autograd/functions/utils.h"
#include "torch/csrc/utils/auto_gpu.h"

#include <memory>
#include <utility>

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
  return wrap_outputs(inputs, std::move(outputs), [&](function_list&& next_functions) {
    return std::make_shared<Error>(msg, std::move(next_functions));
  });
};

}} // namespace torch::autograd
