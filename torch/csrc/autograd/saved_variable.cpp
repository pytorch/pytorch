#include <torch/csrc/autograd/saved_variable.h>

#include <torch/csrc/autograd/anomaly_mode.h>
#include <torch/csrc/autograd/edge.h>
#include <torch/csrc/autograd/engine.h>
#include <torch/csrc/autograd/function.h>
#include <torch/csrc/autograd/grad_mode.h>
#include <torch/csrc/autograd/variable.h>

#include <ATen/Tensor.h>

#include <cstdint>
#include <list>
#include <memory>
#include <sstream>

namespace torch {
namespace autograd {

SavedVariable::SavedVariable(
    const Variable& variable,
    bool is_output,
    bool is_inplace_on_view) {
  if (variable.defined()) {
    // Note [Inference tensor cannot be saved for backward]
    // Invariant:
    //   You can't save an inference tensor for backwards.
    // If an inference tensor was saved for backward in an autograd session and
    // then you reenter inference mode and make an inplace update to the tensor
    // without bumping version_counter, it'll lead to silent wrong result when
    // you do backward() for the previous autograd session.  Technically we
    // don't have to check here since it'll fail when querying `current_version`
    // on the inference tensor, but we can give a much better error message
    // here.
    //
    // Note in the documentation we say "inference tensor cannot participate
    // in autograd" which is more restrictive than the invariant.  In practice
    // the check is more permissive and only error out when an inference tensor
    // is saved for backward.  Whether a tensor is saved for backward is
    // determined by derivative formula and thus varies op by op, so by saying
    // "no inference tensor in autograd" it's easier for users to understand and
    // follow.
    TORCH_CHECK(
        !variable.is_inference(),
        "Inference tensors cannot be saved for backward. To work around "
        "you can make a clone to get a normal tensor and use it in autograd.")

    was_default_constructed_ = false;
    const auto& version_counter = impl::version_counter(variable);
    saved_version_ = version_counter.current_version();
    is_leaf_ = variable.is_leaf();
    is_output_ = is_output;
    is_inplace_on_view_ = is_inplace_on_view;

    if (is_inplace_on_view) {
      TORCH_INTERNAL_ASSERT(!is_leaf_ && is_output);
      weak_grad_fn_ = variable.grad_fn();
    }

    auto maybe_hooks = get_default_hooks();

    // Avoid wrapped numbers from being leaked to the user
    if (maybe_hooks && !variable.unsafeGetTensorImpl()->is_wrapped_number()) {
      save_metadata(variable);
      set_hooks_and_pack_data(std::move(maybe_hooks), variable);
      return;
    }

    // If the variable is a leaf or is not an output, we can safely save the
    // original variable without running the risk of reference cycles.
    // 1. If the variable is not an output, its grad_fn has already been fully
    // created and in particular will be a different Node than the one
    // we are currently constructing (the one that owns this SavedVariable).
    // 2. If the variable is a leaf, it only has weak reference to the
    // grad_accumulator which cannot create a cycle. In those cases, we save the
    // original variable and don't need further processing.
    if (!is_output || is_leaf_) {
      saved_original_ = true;
      data_ = variable;
      return;
    }

    save_metadata(variable);

    // Only do this if we actually need to.
    data_ = variable.tensor_data();
  }
}

void SavedVariable::save_metadata(const Variable& data) {
  // Save output number, version counter and fw_grad if needed

  output_nr_ = data.output_nr();
  version_counter_ = impl::version_counter(data);

  if (is_leaf_) {
    grad_accumulator_ = impl::grad_accumulator(data);
    requires_grad_ = data.requires_grad();
  } else if (!is_output_) {
    grad_fn_ = data.grad_fn();
  }

  // TODO(albanD) This needs to be updated when moving to multiple levels
  const auto& fw_grad = data._fw_grad(/* level */ 0);
  if (fw_grad.defined()) {
    fw_grad_ = std::make_shared<ForwardGrad>();
    fw_grad_->set_value(fw_grad, /* level */ 0);
  }
}

std::unique_ptr<SavedVariableHooks> SavedVariable::get_default_hooks() {
  return Engine::get_default_engine().get_default_saved_variable_hooks();
}

void SavedVariable::reset_data() {
  hooks_.reset();
  grad_fn_.reset();
  data_.reset();
}

SavedVariable::SavedVariable(
    const c10::optional<Variable>& variable,
    bool is_output,
    bool is_inplace_on_view)
    : SavedVariable(
          variable.has_value() ? *variable : Variable(),
          is_output,
          is_inplace_on_view) {}

Variable SavedVariable::unpack(std::shared_ptr<Node> saved_for) const {
  if (was_default_constructed_) {
    return Variable();
  }

  if (!data_.defined()) {
    TORCH_CHECK(hooks_, ERR_BACKWARD_TWICE);
  }

  // We want grad_fn here to provide the most helpful debug message to the user
  // if versions don't match

  auto grad_fn = is_inplace_on_view_ ? weak_grad_fn_.lock()
      : !hooks_ ? saved_original_ ? data_.grad_fn() : nullptr
                : grad_fn_;

  if (!is_leaf_ && !grad_fn) {
    // This issue was introduced when we added logic to save the original
    // because now we rely on data_.grad_fn(), but can be unreliable if the
    // autograd_meta of that saved tensor is cleared with an in-place detach.
    // As a simple fix, we choose to disallow that behavior here even though
    // it makes behavior inconsistent depending on whether you are saving
    // input or output.
    TORCH_CHECK(
        saved_for,
        "Trying to use a saved tensor that has been detached in-place, i.e. with .detach_()."
        "This is not supported, please use out-of-place `.detach()` instead");
    grad_fn = std::move(saved_for);
  }

  // Only check version counter in the case without hooks
  // If user provides hooks, we can't track versions through the hooks
  if (!hooks_) {
    auto current_version = saved_original_
        ? impl::version_counter(data_).current_version()
        : version_counter_.current_version();

    if (saved_version_ != current_version) {
      std::stringstream message;
      message
          << "one of the variables needed for gradient computation has been "
             "modified by an inplace operation: ["
          << data_.toString() << " ";
      if (data_.is_nested()) {
        message << data_._nested_tensor_size() << "]";
      } else {
        message << data_.sizes() << "]";
      }
      if (grad_fn) {
        message << ", which is output " << output_nr_ << " of "
                << grad_fn->name() << ",";
      }
      message << " is at version " << current_version << "; expected version "
              << saved_version_ << " instead.";
      if (!AnomalyMode::is_enabled()) {
        message << " Hint: enable anomaly detection to find the operation "
                   "that failed to compute its gradient, with torch.autograd."
                   "set_detect_anomaly(True).";
      } else {
        message
            << " Hint: the backtrace further above shows the operation "
               "that failed to compute its gradient. The variable in question "
               "was changed in there or anywhere later. Good luck!";
      }
      TORCH_CHECK(false, message.str());
    }
  }

  // The version counter is correct.
  // Additionally, if we deal with a non-leaf variable, we have its correct
  // grad_fn.

  // If we have the original variable, we simply return it
  if (!hooks_ && saved_original_) {
    return data_;
  }

  const auto data = hooks_ ? hooks_->call_unpack_hook() : data_;

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

void SavedVariable::set_hooks_and_pack_data(
    std::unique_ptr<SavedVariableHooks>&& hooks,
    const Variable& data) {
  hooks_ = std::move(hooks);
  at::NoGradGuard guard;
  const auto version = impl::version_counter(data).current_version();
  hooks_->call_pack_hook(saved_original_ ? data.detach() : data);
  TORCH_CHECK(
      version == impl::version_counter(data).current_version(),
      "A saved tensor pack hook is modifying its input in place. "
      "Tensors provided as input to pack hook can not be modified by "
      "in-place operations as this can lead to unexpected side-effects. "
      "Please open an issue if you need to perform in-place operations on "
      "the input to a pack hook.");
}

void SavedVariable::register_hooks(
    std::unique_ptr<SavedVariableHooks>&& hooks) {
  TORCH_INTERNAL_ASSERT(hooks);
  TORCH_CHECK(
      !hooks_,
      "Calling register_hooks on a saved tensor whose hooks have already been set. "
      "Hint: only one pair of hooks is allowed at a time.");
  if (!data_.defined()) {
    if (!was_default_constructed_) {
      TORCH_CHECK(
          false,
          "Calling register_hooks on a saved tensor after it has been freed. "
          "Saved intermediate values of the graph are freed when you call "
          ".backward() or autograd.grad(). Specify retain_graph=True if you "
          "need to backward through the graph a second time or if you need to "
          "access saved variables after calling backward.");
    } else {
      TORCH_CHECK(
          false,
          "Calling register_hooks on a saved tensor with value None is forbidden");
    }
  }
  // If we didn't save the original variable, we already saved metadata
  if (saved_original_) {
    save_metadata(data_);
  }
  set_hooks_and_pack_data(std::move(hooks), data_);
  data_.reset();
}

const char* ERR_BACKWARD_TWICE =
    "Trying to backward through the graph a second time (or directly access saved "
    "tensors after they have already been freed). Saved intermediate values "
    "of the graph are freed when you call .backward() or autograd.grad(). Specify "
    "retain_graph=True if you need to backward through the graph a second time or "
    "if you need to access saved tensors after calling backward.";

} // namespace autograd
} // namespace torch
