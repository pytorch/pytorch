#include <torch/csrc/autograd/saved_variable.h>

#include <torch/csrc/autograd/edge.h>
#include <torch/csrc/autograd/function.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/autograd/anomaly_mode.h>
#include <torch/csrc/autograd/grad_mode.h>

#include <ATen/Tensor.h>

#include <cstdint>
#include <list>
#include <memory>
#include <sstream>

namespace torch { namespace autograd {

SavedVariable::SavedVariable(const Variable& variable, bool is_output, bool is_inplace_on_view) {
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
    TORCH_CHECK(!variable.is_inference(),
      "Inference tensors cannot be saved for backward. To work around "
      "you can make a clone to get a normal tensor and use it in autograd.")

    was_default_constructed_ = false;
    const auto& version_counter = impl::version_counter(variable);
    saved_version_ = version_counter.current_version();
    is_leaf_ = variable.is_leaf();
    is_output_ = is_output;

    // If the variable is a leaf or is not an output, we can safely save the
    // original variable without running the risk of reference cycles.
    // 1. If the variable is not an output, its grad_fn has already been fully
    // created and in particular will be a different Node than the one
    // we are currently constructing (the one that owns this SavedVariable).
    // 2. If the variable is a leaf, it only has weak reference to the grad_accumulator
    // which cannot create a cycle.
    // In those cases, we save the original variable and don't need further processing.
    if (!is_output || is_leaf_) {
      saved_original_ = true;
      data_ = variable;
      return;
    }

    // From now on, we can assume the variable is not a leaf and is an output.

    is_inplace_on_view_ = is_inplace_on_view;
    output_nr_ = variable.output_nr();
    version_counter_ = version_counter;

    // These copies are all shared_ptr copies, so slightly more expensive.
    // Do them here instead of in the init list in case data is undefined.
    data_ = variable.tensor_data();

    if(is_inplace_on_view) {
      weak_grad_fn_ = variable.grad_fn();
    }

    // TODO(albanD) This needs to be updated when moving to multiple levels
    const auto& fw_grad = variable._fw_grad(/* level */ 0);
    if (fw_grad.defined()) {
      fw_grad_ = std::make_shared<ForwardGrad>();
      fw_grad_->set_value(fw_grad, /* level */ 0);
    }
  }
}

void SavedVariable::reset_data() {
  hooks_.reset();
  grad_fn_.reset();
  data_.reset();
}

SavedVariable::SavedVariable(const c10::optional<Variable>& variable, bool is_output, bool is_inplace_on_view)
  : SavedVariable(variable.has_value() ? *variable : Variable(), is_output, is_inplace_on_view) {}

Variable SavedVariable::unpack(std::shared_ptr<Node> saved_for) const {
  if (was_default_constructed_) {
    return Variable();
  }

  if (!data_.defined()) {
    TORCH_CHECK(hooks_, ERR_BACKWARD_TWICE);
  }

  // We want grad_fn here to provide the most helpful debug message to the user
  // if versions don't match

  // TODO(varal7): delete this comment
  // auto grad_fn = !hooks_ ? (saved_original_ ? data_.grad_fn()
  //                                           : is_inplace_on_view_ ? weak_grad_fn_.lock()
  //                                                                 : nullptr)
  //                        : (is_inplace_on_view_ ? weak_grad_fn_.lock() : grad_fn_);

  auto grad_fn = is_inplace_on_view_ ? weak_grad_fn_.lock()
                                     : !hooks_ ? saved_original_ ? data_.grad_fn() : nullptr
                                               : grad_fn_;

  if (!is_leaf_ && !grad_fn) {
    TORCH_CHECK(saved_for, "No grad_fn for non-leaf saved variable");
    grad_fn = std::move(saved_for);
  }

  // Only check version counter in the case without hooks
  // If user provides hooks, we can't track versions through the hooks
  if (!hooks_) {
    auto current_version = saved_original_ ? impl::version_counter(data_).current_version()
                                           : version_counter_.current_version();

    if (saved_version_ != current_version) {
      std::stringstream message;
      message << "one of the variables needed for gradient computation has been "
          "modified by an inplace operation: [" << data_.toString() << " "
          << data_.sizes() << "]";
      if (grad_fn) {
          message << ", which is output " << output_nr_
              << " of " << grad_fn->name() << ",";
      }
      message << " is at version " << current_version
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
      TORCH_CHECK(false, message.str());
    }
  }

  // The version counter is correct.
  // Additionnally, if we deal with a non-leaf variable, we have its correct grad_fn.

  // If we have the original variable, we simply return it
  if (!hooks_ && saved_original_) {
    return data_;
  }

  const auto data = hooks_ ? hooks_->call_unpack_hook(py_data_) : data_;

  // NB: saved views are unpacked as normal Variables (not views) even though
  // they still share the same storage. This works only because we never call
  // in-place functions on unpacked variables.
  Variable var;
  if (grad_fn) {
    var = make_variable(data, Edge(std::move(grad_fn), output_nr_));
  } else {
    var = make_variable(data, requires_grad_);
  }

  impl::set_version_counter(var, version_counter_);

  // If a Variable is a leaf (no grad_fn saved), and it requires_grad, then we
  // should have saved the grad accumulator. Even if the Variable is no longer
  // alive, the accumulator should be kept alive by the references in the
  // graph.
  if (requires_grad_ && !var.grad_fn() && grad_accumulator_.expired()) {
    TORCH_CHECK(false, "No grad accumulator for a saved leaf!");
  }
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

void SavedVariable::register_hooks(std::unique_ptr<SavedVariableHooks>&& hooks) {
  if (!data_.defined()) {
    if (!was_default_constructed_) {
      TORCH_CHECK(false,
        "Calling register_hooks on a saved tensor after it has been freed. "
        "Saved intermediate values of the graph are freed when you call "
        ".backward() or autograd.grad(). Specify retain_graph=True if you "
        "need to backward through the graph a second time or if you need to "
        "access saved variables after calling backward.");
    } else {
      TORCH_CHECK(false,
        "Calling register_hooks on a saved tensor with value None is forbidden");
    }
  }
  TORCH_CHECK(!hooks_,
    "Calling register_hooks on a saved tensor whose hooks have already been set.");
  hooks_ = std::move(hooks);

  // If we didn't save the original variable, we already have all we need to reconstruct it
  if (saved_original_) {
    // TODO(varal7) refactor the next 10 lines
    output_nr_ = data_.output_nr();
    version_counter_ = impl::version_counter(data_);

    // TODO(albanD) This needs to be updated when moving to multiple levels
    const auto& fw_grad = data_._fw_grad(/* level */ 0);
    if (fw_grad.defined()) {
      fw_grad_ = std::make_shared<ForwardGrad>();
      fw_grad_->set_value(fw_grad, /* level */ 0);
    }

    if (is_leaf_) {
      grad_accumulator_ = impl::grad_accumulator(data_);
      requires_grad_ = data_.requires_grad();
    } else if (!is_output_) {
      grad_fn_ = data_.grad_fn();
    } else {
      // Current code assumes that the original variable is saved if and only if (is_leaf_ || !is_output)
      TORCH_INTERNAL_ASSERT(false);
    }
  }

  const bool prev = GradMode::is_enabled();
  GradMode::set_enabled(false);
  py_data_ = hooks_->call_pack_hook(data_);
  GradMode::set_enabled(prev);
  data_.reset();
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
const char* ERR_BACKWARD_TWICE =
    "Trying to backward through the graph a second time (or directly access saved "
    "tensors after they have already been freed). Saved intermediate values "
    "of the graph are freed when you call .backward() or autograd.grad(). Specify "
    "retain_graph=True if you need to backward through the graph a second time or "
    "if you need to access saved tensors after calling backward.";

}} // namespace torch::autograd
