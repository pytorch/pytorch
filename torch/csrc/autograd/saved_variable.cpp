#include <torch/csrc/autograd/saved_variable.h>

#include <torch/csrc/autograd/edge.h>
#include <torch/csrc/autograd/function.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/autograd/anomaly_mode.h>

#include <ATen/Tensor.h>

#include <cstdint>
#include <list>
#include <memory>
#include <sstream>

namespace torch { namespace autograd {

SavedVariable::SavedVariable(const Variable& variable, bool is_output, bool is_inplace_view) {
  if (variable.defined()) {
    // Note [Inference tensor cannot be saved for backward]
    // Invariant:
    //   You can't save an inference tensor for backwards.
    // If an inference tensor was saved for backward in an autograd session and
    // then you reenter inference mode and make an inplace update to the tensor
    // without bumping version_counter, it'll lead to silent wrong result when
    // you do backward() for the previous autograd session.  Technically we don't
    // have to check here since it'll fail when querying `current_version` on
    // the inference tensor, but we can give a much better error message here.
    //
    // Note in the documentation we say "inference tensor cannot participate
    // in autograd" which is more restrictive than the invariant.  In practice
    // the check is more permissive and only error out when an inference tensor
    // is saved for backward.  Whether a tensor is saved for backward is determined
    // by derivative formula and thus varies op by op, so by saying "no inference
    // tensor in autograd" it's easier for users to understand and follow.
    TORCH_CHECK(!variable.unsafeGetTensorImpl()->is_inference_tensor(),
      "Inference tensors cannot be saved for backward. To work around "
      "you can make a clone to get a normal tensor and use it in autograd.")

    was_default_constructed_ = false;
    output_nr_ = variable.output_nr();
    requires_grad_ = variable.requires_grad();
    has_grad_fn_ = !variable.is_leaf();
    is_inplace_view_ = is_inplace_view;
    // These copies are all shared_ptr copies, so slightly more expensive.
    // Do them here instead of in the init list in case data is undefined.
    data_ = variable.tensor_data();
    // TODO(albanD) This needs to be updated when moving to multiple levels
    const auto& fw_grad = variable._fw_grad(/* level */ 0);
    if (fw_grad.defined()) {
      fw_grad_ = std::make_shared<ForwardGrad>();
      fw_grad_->set_value(fw_grad, /* level */ 0);
    }
    if (variable.is_leaf()) {
      grad_accumulator_ = impl::grad_accumulator(variable);
    } else if (!is_output) {
      grad_fn_ = variable.grad_fn();
    } else if (is_inplace_view) {
      weak_grad_fn_ = variable.grad_fn();
    }
    version_counter_ = impl::version_counter(variable);
    saved_version_ = version_counter_.current_version();
  }
}

SavedVariable::SavedVariable(const c10::optional<Variable>& variable, bool is_output, bool is_inplace_view)
  : SavedVariable(variable.has_value() ? *variable : Variable(), is_output, is_inplace_view) {}

Variable SavedVariable::unpack(std::shared_ptr<Node> saved_for) const {
  if (!data_.defined()) {
    if (!was_default_constructed_) {
      throw std::runtime_error(ERR_BACKWARD_TWICE);
    }
    return Variable();
  }

  auto grad_fn = is_inplace_view_ ? weak_grad_fn_.lock() : grad_fn_;
  if (has_grad_fn_ && !grad_fn) {
    if (!saved_for) {
      // If saving the grad_fn would create a circular reference, then it must
      // be passed in to the unpack function.
      throw std::runtime_error("No grad_fn for non-leaf saved variable");
    }
    grad_fn = std::move(saved_for);
  }

  if (saved_version_ != version_counter_.current_version()) {
    std::stringstream message;
    message << "one of the variables needed for gradient computation has been "
        "modified by an inplace operation: [" << data_.toString() << " "
        << data_.sizes() << "]";
    if (grad_fn) {
        message << ", which is output " << output_nr_
            << " of " << grad_fn->name() << ",";
    }
    message << " is at version " << version_counter_.current_version()
        << "; expected version " << saved_version_ << " instead.";
    if (!AnomalyMode::is_enabled()) {
        message << " Hint: enable anomaly detection to find the operation "
            "that failed to compute its gradient, with torch.autograd."
            "set_detect_anomaly(True).";
    }
    else {
        message << " Hint: the backtrace further above shows the operation "
            "that failed to compute its gradient. The variable in question "
            "was changed in there or anywhere later. Good luck!";
    }
    throw std::runtime_error(message.str());
  }

  // NB: saved views are unpacked as normal Variables (not views) even though
  // they still share the same storage. This works only because we never call
  // in-place functions on unpacked variables.
  Variable var;
  if (grad_fn) {
    var = make_variable(data_, Edge(std::move(grad_fn), output_nr_));
  } else {
    var = make_variable(data_, requires_grad_);
  }
  impl::set_version_counter(var, saved_version_);

  // If a Variable is a leaf (no grad_fn saved), and it requires_grad, then we
  // should have saved the grad accumulator. Even if the Variable no longer
  // alive, the accumulator should be kept alive by the references in the
  // graph).
  if (requires_grad_ && !var.grad_fn() && grad_accumulator_.expired())
    throw std::logic_error("No grad accumulator for a saved leaf!");
  impl::set_grad_accumulator(var, grad_accumulator_);

  // NB: var here is never a view so there is no need to make anything special
  // for the case where the saved Tensor was a view. This whole argument relies
  // on the fact that the Tensor returned by this function is never
  // modified in-place.
  if (fw_grad_ && !fw_grad_->empty()) {
    // TODO(albanD) This needs to be updated when moving to multiple levels
    auto new_fw_grad = fw_grad_->value(/* level */ 0);
    var._set_fw_grad(new_fw_grad, /* level */ 0, /* is_inplace_op */ false);
  }

  return var;
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
const char* ERR_BACKWARD_TWICE =
    "Trying to backward through the graph a second time (or directly access saved "
    "variables after they have already been freed). Saved intermediate values "
    "of the graph are freed when you call .backward() or autograd.grad(). Specify "
    "retain_graph=True if you need to backward through the graph a second time or "
    "if you need to access saved variables after calling backward.";

}} // namespace torch::autograd
