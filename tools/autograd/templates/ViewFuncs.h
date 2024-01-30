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
  virtual std::vector<c10::SymInt> get_symints() = 0;
  /// Sets any SymInts in the saved state.
  virtual void set_symints(const std::vector<c10::SymInt>&) = 0;
  /// Returns any Tensors in the saved state.
  virtual std::vector<at::Tensor> get_tensors() = 0;
  /// Sets any Tensors in the saved state.
  virtual void set_tensors(const std::vector<at::Tensor>&) = 0;
  /// Reapplies the view on the given base using the saved state.
  virtual at::Tensor operator()(const at::Tensor&) = 0;
};

struct ChainedViewFunc : public ViewFunc {
  ChainedViewFunc(
      const std::shared_ptr<ViewFunc>&,
      const std::shared_ptr<ViewFunc>&);
  virtual ~ChainedViewFunc() override {};
  virtual std::vector<c10::SymInt> get_symints() override;
  virtual void set_symints(const std::vector<c10::SymInt>&) override;
  virtual std::vector<at::Tensor> get_tensors() override;
  virtual void set_tensors(const std::vector<at::Tensor>&) override;
  virtual at::Tensor operator()(const at::Tensor&) override;

private:
  std::shared_ptr<ViewFunc> first;
  int64_t num_first_symints = -1;
  int64_t num_first_tensors = -1;

  std::shared_ptr<ViewFunc> second;
  int64_t num_second_symints = -1;
  int64_t num_second_tensors = -1;
};

struct ErroringViewFunc : public ViewFunc {
  ErroringViewFunc(const std::string& error_msg) : error_msg(error_msg) {}
  virtual ~ErroringViewFunc() override {};
  virtual std::vector<c10::SymInt> get_symints() override {
    return {};
  }
  virtual void set_symints(const std::vector<c10::SymInt>&) override {}
  virtual std::vector<at::Tensor> get_tensors() override {
    return {};
  }
  virtual void set_tensors(const std::vector<at::Tensor>&) override {}
  virtual at::Tensor operator()(const at::Tensor&) override {
    TORCH_CHECK(false, error_msg);
    return at::Tensor();
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
