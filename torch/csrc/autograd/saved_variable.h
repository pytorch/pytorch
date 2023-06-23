#pragma once

#include <torch/csrc/Export.h>
#include <torch/csrc/autograd/forward_grad.h>
#include <torch/csrc/autograd/saved_variable_hooks.h>

#include <ATen/core/Tensor.h>

#include <cstdint>
#include <memory>

namespace torch {
namespace autograd {

using Variable = at::Tensor;
struct Node;

TORCH_API extern const char* ERR_BACKWARD_TWICE;

/// A snapshot of a variable at a certain version. A `SavedVariable` stores
/// enough information to reconstruct a variable from a certain point in time.
class TORCH_API SavedVariable {
 public:
  SavedVariable() = default;
  SavedVariable(
      const Variable& variable,
      bool is_output,
      bool is_inplace_on_view = false);
  SavedVariable(
      const c10::optional<Variable>& variable,
      bool is_output,
      bool is_inplace_on_view = false);
  SavedVariable(SavedVariable&&) = default;
  SavedVariable& operator=(SavedVariable&&) = default;
  ~SavedVariable() {
    if (fw_grad_) {
      // See note [ Using ForwardGrad ]
      fw_grad_->clear();
    }
  }

  /// Reconstructs the saved variable. Pass `saved_for` as the gradient
  /// function if constructing the `SavedVariable` with it would have caused a
  /// circular reference.
  Variable unpack(std::shared_ptr<Node> saved_for = nullptr) const;

  void register_hooks(std::unique_ptr<SavedVariableHooks>&& hooks);

  void reset_data();

 private:
  // This field contains either:
  // 1. the variable to save
  // 2. or its tensor_data.
  // If storing the variable itself would create a circular reference,
  // we fall into the second case and its metadata is also saved separately.
  // In that case, the grad_fn must be passed in to the unpack function when
  // reconstructing the Variable (except when we are doing an inplace operation
  // on a view, see below). The field saved_original_ below reflects the two
  // cases: its value is true in the first case and false in the second case.
  // The value data_.defined() can be false in three cases:
  // 1. SavedVariable was constructed without a Tensor (the value to save is
  // None), in that case was_default_constructed_ will be kept at true
  // 2. The saved variable has been released by calling
  // SavedVariable::reset_data(), typically during the backward pass
  // 3. Hooks have been registered. In that case, hooks_ will be defined
  // instead. Note that the value of saved_original_ only reflects what happened
  // during the construction of the SavedVariable. If saved_original_ is true,
  // we saved the original tensor in data_, but if the user registers hooks, we
  // will no longer have it (despite the saved_original_ still being true)
  at::Tensor data_;

  // This field is used to store the forward AD gradients associated with
  // the saved Tensor. Note that this shared_ptr must never be shared with
  // either the saved Tensor or the unpacked Tensor. See note [ Using
  // ForwardGrad ]
  std::shared_ptr<ForwardGrad> fw_grad_;

  // Weak version of grad_fn_ that prevents leaks in rebase_history() for
  // inplace views.
  // This variable is used when the user chooses to create a SavedVariable with
  // is_inplace_on_view = true.
  // In that case, the grad_fn passed in to the unpack function at unwrapping
  // time is unused.
  std::weak_ptr<Node> weak_grad_fn_;
  c10::VariableVersion version_counter_;

  uint32_t saved_version_ = 0;
  uint32_t output_nr_ = 0;
  bool was_default_constructed_ = true;
  bool is_inplace_on_view_ = false;
  bool saved_original_ = false;
  bool is_leaf_ = false;
  bool is_output_ = false;

  // Hooks are a pair of functions pack_hook/unpack_hook that provides
  // fine-grained control over how the SavedVariable should save its data.
  // pack_hook is called upon registration, while unpack_hook is called when
  // unpacking.
  std::unique_ptr<SavedVariableHooks> hooks_;
  // Fields grad_fn_, grad_accumulator_, and requires_grad_ are only used if
  // hooks are defined. They are set before pack_hook is called and used after
  // unpack_hook is called.
  std::shared_ptr<Node> grad_fn_;
  // For the usual case where leaf tensors are the input, we expect its
  // grad_acc to be kept alive by the graph. The reason SavedVariable holds
  // a owning reference is to support the case where a custom autograd Function
  // saves an intermediate.
  std::shared_ptr<Node> grad_accumulator_;
  bool requires_grad_ = false;

  void save_metadata(const Variable& data);
  static std::unique_ptr<SavedVariableHooks> get_default_hooks();
  void set_hooks_and_pack_data(
      std::unique_ptr<SavedVariableHooks>&& hooks,
      const Variable& data);
};
} // namespace autograd
} // namespace torch
