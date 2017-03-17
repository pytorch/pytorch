#include "torch/csrc/autograd/variable.h"

#include "torch/csrc/autograd/functions/accumulate_grad.h"
#include "torch/csrc/utils/auto_gpu.h"

using namespace torch;
using namespace thpp;

namespace torch { namespace autograd {

Variable::Variable(
  std::unique_ptr<thpp::Tensor> data,
  bool requires_grad,
  bool is_volatile)
    : data(std::move(data))
    , grad_fn(nullptr)
    , grad(nullptr)
    , version_counter(new VariableVersion())
    , requires_grad(requires_grad)
    , is_volatile(is_volatile)
    , output_nr(0)
    , pyobj(nullptr)
{
  if (!this->data) throw std::runtime_error("Variable data is NULL");
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
    , output_nr(grad_fn->num_inputs++)
    , pyobj(nullptr)
{
  if (!this->data) throw std::runtime_error("Variable data is NULL");
}

auto Variable::get_grad_accumulator() -> std::shared_ptr<Function> {
  using weak_type = std::weak_ptr<Function>;

  static std::shared_ptr<Function> null_shared_ptr;
  static weak_type null_weak_ptr;

  if (!requires_grad) return std::shared_ptr<Function>();

  auto result = grad_accumulator.lock();
  if (result) return result;

  // That didn't work, we need to allocate it, but taking into account that other
  // threads might be doing the same thing.
  std::lock_guard<std::mutex> lock(grad_accumulator_lock);

  result = grad_accumulator.lock();
  if (result) return result;

  result = std::make_shared<AccumulateGrad>(shared_from_this());
  grad_accumulator = result;
  return result;
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
