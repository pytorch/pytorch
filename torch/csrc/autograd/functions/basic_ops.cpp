#include <torch/csrc/autograd/functions/basic_ops.h>

#include <torch/csrc/autograd/function.h>
#include <torch/csrc/autograd/functions/utils.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/dynamo/compiled_autograd.h>

#include <ATen/ATen.h>

#include <memory>
#include <utility>

namespace torch {
namespace autograd {

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
    outputs.emplace_back(
        var.defined() ? var.clone().tensor_data() : at::Tensor());
  }
  return wrap_outputs(inputs, std::move(outputs), [&](edge_list&& next_edges) {
    return std::make_shared<UndefinedGradBackward>(std::move(next_edges));
  });
}

auto UndefinedGradBackward::apply(variable_list&& output_grads)
    -> variable_list {
  tensor_list input_grads;
  output_grads.reserve(input_grads.size());
  for (auto& grad : output_grads) {
    (void)grad; // Suppress unused variable warning
    input_grads.emplace_back();
  }
  return input_grads;
}

auto Identity::apply(variable_list&& grads) -> variable_list {
  return std::move(grads);
}

void GraphRoot::compiled_args(CompiledNodeArgs& args) {
  args.collect(outputs);
}
variable_list GraphRoot::apply_with_saved(
    const variable_list& inputs,
    SwapSavedVariables& saved) {
  saved.before(outputs);
  variable_list result(outputs);
  saved.after(outputs);
  return result;
}

} // namespace autograd
} // namespace torch
