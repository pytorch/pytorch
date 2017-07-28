#include "torch/csrc/autograd/variable.h"

#include "torch/csrc/autograd/functions/accumulate_grad.h"
#include "torch/csrc/utils/auto_gpu.h"

using namespace torch;

namespace torch { namespace autograd {

Variable::Variable(
  at::Tensor data,
  bool requires_grad,
  bool is_volatile)
    : data(data)
    , grad_fn(nullptr)
    , grad(nullptr)
    , version_counter(new VariableVersion())
    , requires_grad(requires_grad)
    , is_volatile(is_volatile)
    , output_nr(0)
    , pyobj(nullptr)
{
  if (!this->data.defined()) {
    throw std::runtime_error("Variable data is NULL");
  }
}

Variable::Variable(
  at::Tensor data,
  std::shared_ptr<Function> grad_fn)
    : data(data)
    , grad_fn(grad_fn)
    , grad(nullptr)
    , version_counter(new VariableVersion())
    , requires_grad(grad_fn->is_executable)
    , is_volatile(false)
    , output_nr(grad_fn->num_inputs++)
    , pyobj(nullptr)
{
  if (!this->data.defined()) {
    throw std::runtime_error("Variable data is NULL");
  }
}

auto Variable::get_grad_accumulator() -> std::shared_ptr<Function> {
  if (grad_fn) {
    throw std::logic_error("get_grad_accumulator() should be only called on leaf Variables");
  }
  if (!requires_grad) return nullptr;

  std::lock_guard<std::mutex> lock(grad_accumulator_lock);

  auto result = grad_accumulator.lock();
  if (result) return result;

  result = std::make_shared<AccumulateGrad>(shared_from_this());
  grad_accumulator = result;
  return result;
}

auto SavedVariable::unpack(std::shared_ptr<Function> saved_for) -> std::shared_ptr<Variable> {
  if (!data.defined()) {
    if (version) {
      throw std::runtime_error(ERR_BACKWARD_TWICE);
    }
    return nullptr;
  }

  int current_version = **version;
  if (expected_version != current_version) {
    throw std::runtime_error("one of the variables "
        "needed for gradient computation has been modified by an "
        "inplace operation");
  }

  auto new_var = std::make_shared<Variable>(data, requires_grad, is_volatile);
  if (has_grad_fn && !grad_fn) {
    if (!saved_for) {
      // If saving the grad_fn would create a circular reference, then it must
      // be passed in to the unpack function.
      throw std::runtime_error("No grad_fn for non-leaf saved variable");
    }
    new_var->grad_fn = saved_for;
  } else {
    new_var->grad_fn = grad_fn;
  }
  new_var->version_counter->join_with(*version);
  // If a Variable is a leaf (no grad_fn saved), and it requires_grad, then we
  // should have saved the grad accumulator. Even if the Variable no longer
  // alive, the accumulator should be kept alive by the references in the graph).
  if (requires_grad && !new_var->grad_fn && grad_accumulator.expired())
    throw std::logic_error("No grad accumulator for a saved leaf!");
  new_var->grad_accumulator = grad_accumulator;

  return new_var;
}

const char* ERR_BACKWARD_TWICE =
    "Trying to backward through the graph a second time, but the buffers have "
    "already been freed. Specify retain_graph=True when calling backward "
    "the first time.";

}} // namespace torch::autograd
