#include "torch/csrc/autograd/saved_variable.h"

#include "torch/csrc/autograd/edge.h"
#include "torch/csrc/autograd/function.h"
#include "torch/csrc/autograd/variable.h"

#include <ATen/ATen.h>

#include <cstdint>
#include <list>
#include <memory>

namespace torch { namespace autograd {

SavedVariable::SavedVariable(const Variable& variable, bool is_output) {
  if (variable.defined()) {
    was_default_constructed_ = false;
    output_nr_ = variable.output_nr();
    requires_grad_ = variable.requires_grad();
    has_grad_fn_ = !variable.is_leaf();
    // These copies are all shared_ptr copies, so slightly more expensive.
    // Do them here instead of in the init list in case data is undefined.
    data_ = variable.data();
    if (variable.is_leaf()) {
      grad_accumulator_ = variable.grad_accumulator();
    } else if (!is_output) {
      grad_fn_ = variable.grad_fn();
    }
    version_counter_ = variable.version_counter();
    saved_version_ = version_counter_.current_version();
  }
}

Variable SavedVariable::unpack(std::shared_ptr<Function> saved_for) const {
  auto grad_fn = grad_fn_;
  if (has_grad_fn_ && !grad_fn) {
    // If saving the grad_fn would create a circular reference, then it must
    // be passed in to the unpack function.
    AT_CHECK(saved_for, "No grad_fn for non-leaf saved variable");
    grad_fn = std::move(saved_for);
  }

  if (!data_.defined()) {
    AT_CHECK(was_default_constructed_,
             "In function ", grad_fn->name(), ", ", ERR_BACKWARD_TWICE);
    return Variable();
  }

  AT_CHECK(saved_version_ == version_counter_.current_version(),
           "In function ", grad_fn->name(), ", one of the variables needed ",
           "for gradient computation has been modified by an inplace ",
           "operation");

  // NB: saved views are unpacked as normal Variables (not views) even though
  // they still share the same storage. This works only because we never call
  // in-place functions on unpacked variables.
  Variable var;
  if (grad_fn) {
    var = make_variable(data_, Edge(std::move(grad_fn), output_nr_));
  } else {
    var = make_variable(data_, requires_grad_);
  }
  var.set_version_counter(saved_version_);

  // If a Variable is a leaf (no grad_fn saved), and it requires_grad, then we
  // should have saved the grad accumulator. Even if the Variable no longer
  // alive, the accumulator should be kept alive by the references in the
  // graph).
  AT_ASSERTM(!requires_grad_ || var.grad_fn() || !grad_accumulator_.expired(),
             "In function ", grad_fn->name(), ", can't find grad accumulator ",
             "for a saved leaf!");
  var.set_grad_accumulator(grad_accumulator_);

  return var;
}

const char* ERR_BACKWARD_TWICE =
    "Trying to backward through the graph a second time, but the buffers have "
    "already been freed. Specify retain_graph=True when calling backward "
    "the first time.";

}} // namespace torch::autograd
