#pragma once

// ${generated_comment}

#include <torch/library.h>

#include <c10/core/SymIntArrayRef.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Operators.h>
#else
$ops_headers
#endif

namespace torch { namespace autograd {

// forward declaration so it can be a friend class of ViewFunc
struct ViewFuncSavedStateGuard;

/// Base class for view functions, providing reapplication of a view on a new base.
/// Each view op should get a codegenerated subclass of this class containing
/// any state needed to reconstruct the view. The class also provides convenience
/// accessors for saved SymInts / tensor state. This is useful for e.g. fake-ification,
/// where we want to use symbolic values or fake tensors instead.
struct TORCH_API ViewFunc {
  virtual ~ViewFunc() {}
  /// Returns any SymInts in the saved state.
  virtual std::vector<c10::SymInt> get_symints() const = 0;
  /// Returns any Tensors in the saved state.
  virtual std::vector<at::Tensor> get_tensors() const = 0;
  /// Reapplies the view on the given base using the saved state.
  /// If specified, the provided values for SymInts / Tensors
  /// are hot-swapped in for the view call.
  virtual at::Tensor operator()(
      const at::Tensor&,
      const std::optional<std::vector<c10::SymInt>>& = c10::nullopt,
      const std::optional<std::vector<at::Tensor>>& = c10::nullopt) const = 0;
  // Allow limited setter access to maintain the const-ness invariant.
  friend class ViewFuncSavedStateGuard;

protected:
  /// Sets the values of any SymInts in the saved state. The input vector size must
  /// match the number of SymInts in the saved state (i.e. the size of the list
  /// returned by get_symints()).
  virtual void set_symints(const std::vector<c10::SymInt>&) = 0;
  /// Sets the values of any Tensors in the saved state. The input vector size must
  /// match the number of Tensors in the saved state (i.e. the size of the list
  /// returned by get_tensors()).
  virtual void set_tensors(const std::vector<at::Tensor>&) = 0;
};

// RAII guard for modifying and restoring saved SymInt / tensor state in a
// ViewFunc.
struct ViewFuncSavedStateGuard {
  ViewFuncSavedStateGuard(
      ViewFunc& view_func,
      const std::optional<std::vector<c10::SymInt>>& new_symints,
      const std::optional<std::vector<at::Tensor>>& new_tensors)
      : view_func(view_func) {
    // save old state and set new state
    old_symints = view_func.get_symints();
    old_tensors = view_func.get_tensors();
    if (new_symints.has_value()) {
      view_func.set_symints(*new_symints);
    }
    if (new_tensors.has_value()) {
      view_func.set_tensors(*new_tensors);
    }
  }

  ~ViewFuncSavedStateGuard() {
    // restore previous state
    view_func.set_symints(old_symints);
    view_func.set_tensors(old_tensors);
  }

  ViewFunc& view_func;
  std::vector<c10::SymInt> old_symints;
  std::vector<at::Tensor> old_tensors;
};

struct ChainedViewFunc : public ViewFunc {
  ChainedViewFunc(
      const std::shared_ptr<ViewFunc>&,
      const std::shared_ptr<ViewFunc>&);
  virtual ~ChainedViewFunc() override {};
  virtual std::vector<c10::SymInt> get_symints() const override;
  virtual std::vector<at::Tensor> get_tensors() const override;
  virtual at::Tensor operator()(
      const at::Tensor&,
      const std::optional<std::vector<c10::SymInt>>& = c10::nullopt,
      const std::optional<std::vector<at::Tensor>>& = c10::nullopt) const override;

protected:
  // Purposefully do nothing for these and let the sub-ViewFuncs handle it
  virtual void set_symints(const std::vector<c10::SymInt>&) override {}
  virtual void set_tensors(const std::vector<at::Tensor>&) override {}

private:
  std::shared_ptr<ViewFunc> first;
  // NB: These must be mutable so we can store them during the const getter call
  mutable size_t num_first_symints;
  mutable size_t num_first_tensors;

  std::shared_ptr<ViewFunc> second;
  // NB: These must be mutable so we can store them during the const getter call
  mutable size_t num_second_symints;
  mutable size_t num_second_tensors;
};

struct ErroringViewFunc : public ViewFunc {
  ErroringViewFunc(const std::string& error_msg) : error_msg(error_msg) {}
  virtual ~ErroringViewFunc() override {};
  virtual std::vector<c10::SymInt> get_symints() const override {
    return {};
  }
  virtual std::vector<at::Tensor> get_tensors() const override {
    return {};
  }
  virtual at::Tensor operator()(
      const at::Tensor&,
      const std::optional<std::vector<c10::SymInt>>& = c10::nullopt,
      const std::optional<std::vector<at::Tensor>>& = c10::nullopt) const override {
    TORCH_CHECK(false, error_msg);
    return at::Tensor();
  }

protected:
  virtual void set_symints(const std::vector<c10::SymInt>&) override {}
  virtual void set_tensors(const std::vector<at::Tensor>&) override {}

private:
  std::string error_msg;
};

namespace generated {

using at::Scalar;
using at::Tensor;
using at::IntArrayRef;
using at::ArrayRef;
using at::Type;
using at::ScalarType;
using c10::optional;
using c10::fmap;

${view_func_declarations}

}}} // namespace torch::autograd::generated
