#include "torch/csrc/autograd/variable.h"

#include "torch/csrc/utils/auto_gpu.h"

using namespace torch;
using namespace thpp;

namespace torch { namespace autograd {

Variable::Variable(
  std::unique_ptr<thpp::Tensor> data,
  bool requires_grad,
  bool is_volatile,
  bool is_leaf)
    : data(std::move(data))
    , grad_fn(nullptr)
    , grad(nullptr)
    , version_counter(new VariableVersion())
    , requires_grad(requires_grad)
    , is_volatile(is_volatile)
    , is_leaf(is_leaf)
    , output_nr(0)
    , pyobj(nullptr)
{
  if (!this->data) throw std::runtime_error("Variable data is NULL");
  // Function fields
  this->is_executable = requires_grad;
}

Variable::Variable(
  std::unique_ptr<thpp::Tensor> data,
  std::shared_ptr<Function> grad_fn)
    : data(std::move(data))
    , grad_fn(grad_fn)
    , grad(nullptr)
    , version_counter(new VariableVersion())
    , requires_grad(grad_fn->is_executable)
    , is_volatile(false)
    , is_leaf(false)
    , output_nr(grad_fn->num_inputs++)
    , pyobj(nullptr)
{
  if (!this->data) throw std::runtime_error("Variable data is NULL");
  this->is_executable = this->requires_grad;
}

auto Variable::accumulate_grad(std::shared_ptr<Variable> gradOutput) -> void {
  if (!pre_hooks.empty()) {
    for (auto& hook : pre_hooks) {
      gradOutput = (*hook)(variable_list({gradOutput}))[0];
    }
  }
  AutoGPU auto_gpu(gradOutput->data->getDevice());
  if (!grad) {
    std::unique_ptr<Tensor> data(gradOutput->data->clone());
    grad = std::make_shared<Variable>(std::move(data), false, true);
  } else if (grad->data->isSparse() && !gradOutput->data->isSparse()) {
    auto* sum = gradOutput->data->clone();
    sum->cadd(*sum, *grad->data);
    grad->data.reset(sum);
  } else {
    grad->data->cadd(*grad->data, *gradOutput->data);
  }
}

auto Variable::apply(const variable_list& gradOutputs) -> variable_list {
  if (grad_fn) {
    throw std::logic_error("non-leaf variable saved in a graph!");
  }
  if (**version_counter != 0) {
    throw std::runtime_error("leaf variable was used in an inplace operation");
  }
  if (gradOutputs.size() != 1) {
    throw std::runtime_error(std::string("variable expected 1 grad_output but got ") +
                             std::to_string(gradOutputs.size()));
  }
  accumulate_grad(gradOutputs[0]);
  return variable_list();
}

auto Variable::save() const -> SavedVariable {
  return SavedVariable(
    std::unique_ptr<Tensor>(data->clone_shallow()),
    **version_counter,
    std::unique_ptr<VariableVersion>(version_counter->new_saved_ref()));
}

auto Variable::save_opt(Variable* var) -> SavedVariable {
 return var ? var->save() : SavedVariable();
}

auto SavedVariable::unpack() -> std::unique_ptr<thpp::Tensor>& {
  if (data) {
    int current_version = **version;
    if (expected_version != current_version) {
      throw std::runtime_error("one of the variables "
          "needed for gradient computation has been modified by an "
          "inplace operation");
    }
  }
  return data;
}

}} // namespace torch::autograd
