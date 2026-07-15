#include <torch/csrc/autograd/functions/accumulate_grad.h>

#include <ATen/core/dispatch/Dispatcher.h>
#include <torch/csrc/autograd/functions/basic_ops.h>
#include <torch/csrc/autograd/functions/tensor.h>
#include <torch/csrc/autograd/functions/utils.h>
#include <torch/csrc/autograd/grad_mode.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/dynamo/compiled_autograd.h>

#include <cstdint>
#include <stdexcept>
#include <utility>

namespace torch::autograd {

using torch::dynamo::autograd::IValuePacker;

namespace {

void AccumulateGrad_apply_impl(
    variable_list&& grads,
    at::Tensor& variable,
    at::Tensor& variable_grad,
    int64_t num_expected_refs,
    const std::function<void(at::Tensor&&)>& grad_update,
    std::mutex* mutex = nullptr) {
  check_input_variables("AccumulateGrad", grads, 1, 0);

  if (!grads[0].defined())
    return;
  if (!variable.requires_grad())
    return;

  // std::move(grads[0]) to avoid bumping up refcount
  at::Tensor new_grad = std::move(grads[0]);

  // Acquire lock to here protect thread safety on variable, this ensures
  // AccumulateGrad does not race to shared variable from different threads
  // when updating the gradients. We don't ensure thread safety on hooks
  // and rely on user to provide thread safe hooks
  // see Note [Thread Safety on Autograd Node]
  // need to still lock for eager here
  std::optional<std::lock_guard<std::mutex>> lock;
  if (mutex != nullptr) {
    lock.emplace(*mutex);
  }

  AccumulateGrad::accumulateGrad(
      variable, variable_grad, new_grad, num_expected_refs, grad_update);
}

variable_list AccumulateGrad_apply_functional_no_hooks_ivalue(
    const variable_list& grads,
    const ivalue_list& args) {
  PackedArgs r(args);
  auto variable = r.unpack<at::Tensor>();
  auto variable_grad = r.unpack<at::Tensor>();
  auto has_post_hooks = r.unpack<bool>();

  // Functional Tensors insert an Error node to assert that backward is never
  // called
  if (variable.grad_fn() &&
      std::dynamic_pointer_cast<Error>(variable.grad_fn()) == nullptr) {
    throw std::logic_error(
        "leaf variable has been moved into the graph interior");
  }

  at::Tensor functional_grad;
  AccumulateGrad_apply_impl(
      variable_list(grads),
      variable,
      variable_grad,
      1 + has_post_hooks,
      [&functional_grad](at::Tensor&& grad_update) {
        functional_grad = std::move(grad_update);
      },
      nullptr // no mutex needed since this is executed under a single thread
  );
  if (!functional_grad.defined()) {
    // In-place accumulation (Case 2.3) does not execute grad_update
    functional_grad = std::move(variable_grad);
  }
  return {functional_grad};
}
} // namespace

// AccumulateGrad sets sequence_nr to the max value so it's always called
// ASAP during backwards.
AccumulateGrad::AccumulateGrad(Variable variable_)
    : Node(/*sequence_nr=*/UINT64_MAX), variable(std::move(variable_)) {
  add_input_metadata(variable);
}

// NOLINTNEXTLINE(cppcoreguidelines-rvalue-reference-param-not-moved)
auto AccumulateGrad::apply(variable_list&& grads) -> variable_list {
  if (variable.grad_fn()) {
    throw std::logic_error(
        "leaf variable has been moved into the graph interior");
  }

  at::Tensor& variable_grad = variable.mutable_grad();

  // If the function has post hooks (for example, a DDP allreduce hook),
  // call_function in Engine.cpp will temporarily bump the expected refcount
  // by one, hence the addition of !post_hooks().empty() for
  // 'num_expected_refs' in addition to the one reference that we're holding.
  // 'num_expected_refs' is used to determine whether or not we should clone
  // the grad or can steal the grad.
  AccumulateGrad_apply_impl(
      std::move(grads),
      variable,
      variable_grad,
      1 + !post_hooks().empty() /* num_expected_refs */,
      [&variable_grad](at::Tensor&& grad_update) {
        variable_grad = std::move(grad_update);
      },
      &mutex_);

  auto& hook = tensor_post_acc_grad_hooks();
  if (hook != nullptr) {
    (*hook)(variable);
  }

  return variable_list();
}

void AccumulateGrad::compiled_args(CompiledNodeArgs& args) const {
  if (args.cond(variable.defined() && variable.requires_grad())) {
    args.collect(variable);
    args.collect(variable.grad());
  }
  args.collect(GradMode::is_enabled());
  const auto& hook = tensor_post_acc_grad_hooks();
  if (hook != nullptr) {
    hook->compiled_args(args);
  }
}

variable_list AccumulateGrad::apply_with_saved(
    const variable_list& grads,
    SwapSavedVariables& saved) {
  if (!(variable.defined() && variable.requires_grad()) ||
      !grads[0].defined()) {
    return variable_list();
  }
  TORCH_INTERNAL_ASSERT(!variable.grad_fn() && grads.size() == 1);
  at::Tensor variable_copy = variable;
  at::Tensor grad_copy = variable.grad();
  saved.before(variable_copy);
  saved.before(grad_copy);
  variable_copy.mutable_grad() = grad_copy;

  // name() includes namespace for historical reasons:
  // torch::autograd::AccumulateGrad For Compiled Autograd, we just want the op
  // name without the namespace
  std::string name = "AccumulateGrad";

  // proxy a call to torch.ops.inductor.accumulate_grad_.default
  static bool flag [[maybe_unused]] = [&]() {
    std::vector<at::TypePtr> schema = {
        IValuePacker<at::Tensor>::packed_type(),
        IValuePacker<at::Tensor>::packed_type(),
        IValuePacker<bool>::packed_type()};
    const auto& interface = torch::dynamo::autograd::getPyCompilerInterface();
    interface->bind_function(
        saved.get_py_compiler(),
        name,
        AccumulateGrad_apply_functional_no_hooks_ivalue,
        schema);
    return true;
  }();

  const auto& interface = torch::dynamo::autograd::getPyCompilerInterface();
  interface->call_accumulate_grad(
      saved.get_py_compiler(), variable_copy, grads[0], !post_hooks().empty());

  auto& hook = tensor_post_acc_grad_hooks();
  if (hook != nullptr) {
    hook->apply_with_saved(variable_copy, saved);
  }
  saved.after(variable_copy);
  saved.after(grad_copy);

  return variable_list();
}

} // namespace torch::autograd
