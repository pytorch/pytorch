#include <torch/csrc/autograd/functions/basic_ops.h>

#include <torch/csrc/autograd/function.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/autograd/functions/utils.h>

#include <ATen/ATen.h>

#include <memory>
#include <utility>

namespace torch { namespace autograd {

auto Error::apply(variable_list&& inputs) -> variable_list {
  throw std::runtime_error(msg);
}

auto DelayedError::apply(variable_list&& inputs) -> variable_list {
  tensor_list outputs;
  outputs.reserve(inputs.size());
  for (auto& var : inputs) {
    // FIXME: share version counters
    outputs.emplace_back(var.defined() ? var.tensor_data() : at::Tensor());
  }
  return wrap_outputs(inputs, std::move(outputs), [&](edge_list&& next_edges) {
    return std::make_shared<Error>(msg, std::move(next_edges));
  });
}

auto UndefinedGrad::apply(variable_list&& inputs) -> variable_list {
  tensor_list outputs;
  outputs.reserve(inputs.size());
  for (auto& var : inputs) {
    outputs.emplace_back(var.defined() ? var.clone().tensor_data() : at::Tensor());
  }
  return wrap_outputs(inputs, std::move(outputs), [&](edge_list&& next_edges) {
    return std::make_shared<UndefinedGradBackward>(std::move(next_edges));
  });
}

auto UndefinedGradBackward::apply(variable_list&& output_grads) -> variable_list {
  tensor_list input_grads;
  output_grads.reserve(input_grads.size());
  // NOLINTNEXTLINE(clang-diagnostic-unused-variable)
  for (auto& grad : output_grads) {
    input_grads.emplace_back(at::Tensor());
  }
  return input_grads;
}

auto Identity::apply(variable_list&& grads) -> variable_list {
  return std::move(grads);
}

}} // namespace torch::autograd
