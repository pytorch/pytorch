#pragma once

#include <torch/csrc/WindowsTorchApiMacro.h>

#include <ATen/ATen.h>

#include <cstdint>
#include <memory>

namespace torch { namespace autograd {

struct Variable;
struct Function;

TORCH_API extern const char* ERR_BACKWARD_TWICE;

/// A snapshot of a variable at a certain version. A `SavedVariable` stores
/// enough information to reconstruct a variable from a certain point in time.
class TORCH_API SavedVariable {
 public:
  SavedVariable() = default;
  SavedVariable(const Variable& variable, bool is_output);
  SavedVariable(SavedVariable&&) = default;
  SavedVariable& operator=(SavedVariable&&) = default;

  /// Reconstructs the saved variable. Pass `saved_for` as the gradient
  /// function if constructing the `SavedVariable` with it would have caused a
  /// circular reference.
  Variable unpack(std::shared_ptr<Function> saved_for = nullptr) const;

  void reset_data() {
    return data_.reset();
  }

  void reset_grad_function() {
    grad_fn_.reset();
  }

 private:
  at::Tensor data_;

  // The gradient function associated with this node. If has_grad_fn
  // is false, then this is a leaf node. Note that the grad_fn is not saved if
  // it would create a circular reference. In that case, the grad_fn must be
  // passed in to the unpack function when reconstructing the Variable.
  std::shared_ptr<Function> grad_fn_;
  std::weak_ptr<Function> grad_accumulator_;
  c10::VariableVersion version_counter_;

  uint32_t saved_version_ = 0;
  uint32_t output_nr_ = 0;
  bool was_default_constructed_ = true;
  bool requires_grad_ = false;
  bool has_grad_fn_ = false;
};
}} // namespace torch::autograd
