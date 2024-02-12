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

/// Base class for view functions, providing reapplication of a view on a new base.
/// Each view op should get a codegenerated subclass of this class containing
/// any state needed to reconstruct the view. The class also provides convenience
/// accessors for saved SymInts / tensor state. This is useful for e.g. fake-ification,
/// where we want to use symbolic values or fake tensors instead.
struct TORCH_API ViewFunc {
  virtual ~ViewFunc() {}
  /// Returns any SymInts in the saved state.
  virtual std::vector<c10::SymInt> get_symints() const { return {}; }
  /// Returns the number of SymInts in the saved state.
  virtual size_t num_symints() const { return 0; }
  /// Returns any tensors in the saved state.
  virtual std::vector<at::Tensor> get_tensors() const { return {}; }
  /// Returns the number of tensors in the saved state.
  virtual size_t num_tensors() const { return 0; }
  /// Reapplies the view on the given base using the saved state.
  virtual at::Tensor operator()(const at::Tensor&) const = 0;
  /// Returns a clone of this ViewFunc, optionally with the specified saved state.
  virtual std::unique_ptr<ViewFunc> clone_and_set(
      std::optional<std::vector<c10::SymInt>> = c10::nullopt,
      std::optional<std::vector<at::Tensor>> = c10::nullopt) const = 0;

protected:
  /// Sets the values of any SymInts in the saved state. The input vector size must
  /// match the number of SymInts in the saved state (i.e. the size of the list
  /// returned by get_symints()).
  virtual void set_symints(std::vector<c10::SymInt>) {}
  /// Sets the values of any Tensors in the saved state. The input vector size must
  /// match the number of Tensors in the saved state (i.e. the size of the list
  /// returned by get_tensors()).
  virtual void set_tensors(std::vector<at::Tensor>) {}
};

/// ViewFunc that represents a chain of two ViewFuncs.
struct ChainedViewFunc : public ViewFunc {
  ChainedViewFunc(
      std::unique_ptr<ViewFunc> first,
      std::unique_ptr<ViewFunc> second)
      : first(std::move(first)),
        second(std::move(second)) {}
  virtual ~ChainedViewFunc() override {};
  virtual std::vector<c10::SymInt> get_symints() const override;
  virtual size_t num_symints() const override {
    return first->num_symints() + second->num_symints();
  }
  virtual std::vector<at::Tensor> get_tensors() const override;
  virtual size_t num_tensors() const override {
    return first->num_tensors() + second->num_tensors();
  }
  virtual at::Tensor operator()(const at::Tensor&) const override;
  virtual std::unique_ptr<ViewFunc> clone_and_set(
      std::optional<std::vector<c10::SymInt>> = c10::nullopt,
      std::optional<std::vector<at::Tensor>> = c10::nullopt) const override;

private:
  std::unique_ptr<ViewFunc> first;
  std::unique_ptr<ViewFunc> second;
};

/// ViewFunc that errors with a specified error message when called.
struct ErroringViewFunc : public ViewFunc {
  ErroringViewFunc(const std::string& error_msg) : error_msg(error_msg) {}
  virtual ~ErroringViewFunc() override {};
  virtual at::Tensor operator()(const at::Tensor&) const override {
    TORCH_CHECK(false, error_msg);
  }
  virtual std::unique_ptr<ViewFunc> clone_and_set(
      std::optional<std::vector<c10::SymInt>> = c10::nullopt,
      std::optional<std::vector<at::Tensor>> = c10::nullopt) const override {
    return std::make_unique<ErroringViewFunc>(error_msg);
  }

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
