#include "torch/csrc/autograd/saved_variable.h"

#include "torch/csrc/autograd/function.h"

using namespace at;

namespace torch { namespace autograd {

SavedVariable::SavedVariable(const Variable& variable, Function* saved_for)
  : SavedVariable() {
  if (!variable.defined()) {
    return;
  }
  data = variable.data();
  requires_grad = variable.requires_grad();
  is_volatile = variable.is_volatile();
  expected_version = variable.current_version();
  version = variable.get()->version_counter->new_saved_ref();
  has_grad_fn = variable.grad_fn() != nullptr;
  if (!has_grad_fn) {
    grad_accumulator = variable.grad_accumulator();
  }
  if (variable.grad_fn().get() != saved_for) {
    grad_fn = variable.grad_fn();
  }
  if (variable.tracing_state()) {
    tracing_state.reset(new jit::tracer::ValueTracingState(*variable.tracing_state()));
  }
}

auto SavedVariable::unpack(std::shared_ptr<Function> saved_for) const -> Variable {
  if (!data.defined()) {
    if (version) {
      throw std::runtime_error(ERR_BACKWARD_TWICE);
    }
    return Variable();
  }

  int current_version = **version;
  if (expected_version != current_version) {
    throw std::runtime_error("one of the variables "
        "needed for gradient computation has been modified by an "
        "inplace operation");
  }

  Variable var = make_variable(data, requires_grad, is_volatile);
  if (has_grad_fn && !grad_fn) {
    if (!saved_for) {
      // If saving the grad_fn would create a circular reference, then it must
      // be passed in to the unpack function.
      throw std::runtime_error("No grad_fn for non-leaf saved variable");
    }
    var.grad_fn() = saved_for;
  } else {
    var.grad_fn() = grad_fn;
  }
  var.get()->version_counter->join_with(*version);

  // If a Variable is a leaf (no grad_fn saved), and it requires_grad, then we
  // should have saved the grad accumulator. Even if the Variable no longer
  // alive, the accumulator should be kept alive by the references in the graph).
  if (requires_grad && !var.grad_fn() && grad_accumulator.expired())
    throw std::logic_error("No grad accumulator for a saved leaf!");
  var.get()->grad_accumulator = grad_accumulator;
  if (tracing_state)
    var.tracing_state().reset(new jit::tracer::ValueTracingState(*tracing_state));

  return var;
}

auto SavedVariable::unpack_data(std::shared_ptr<Function> saved_for) const -> Tensor {
  auto var = unpack(saved_for);
  if (var.defined()) {
    return var.data();
  }
  return Tensor();
}


const char* ERR_BACKWARD_TWICE =
    "Trying to backward through the graph a second time, but the buffers have "
    "already been freed. Specify retain_graph=True when calling backward "
    "the first time.";

}} // namespace torch::autograd
