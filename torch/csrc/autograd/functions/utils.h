#pragma once

#include <torch/csrc/Export.h>
#include <torch/csrc/autograd/InferenceMode.h>
#include <torch/csrc/autograd/autograd.h>
#include <torch/csrc/autograd/function.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/utils/variadic.h>

#include <ATen/core/Tensor.h>

#include <functional>
#include <memory>
#include <vector>

namespace torch::autograd {

using function_constructor = std::function<std::shared_ptr<Node>(edge_list&&)>;

/**
 * Wraps the tensor outputs in variables and creates the grad_fn and sets the
 * grad_fn if necessary.
 */
TORCH_API variable_list wrap_outputs(
    const variable_list& inputs,
    tensor_list&& outputs,
    const function_constructor& ctr);

///  Checks that inputs contains exactly `args` items and that the first
///  `required_args`
/// items are not nullptr. If not specified, `required_args` defaults to `args`.
TORCH_API void check_input_variables(
    const char* name,
    const variable_list& inputs,
    int args,
    int required_args = -1,
    bool allow_undefined = false);

struct ComputeRequiresGrad : IterArgs<ComputeRequiresGrad> {
  bool out = false;
  using IterArgs<ComputeRequiresGrad>::operator();
  void operator()(const at::Tensor& tensor) {
    const auto& var = static_cast<const Variable&>(tensor);
    if (var.defined() && var.requires_grad()) {
      out = true;
    }
  }
  void operator()(const std::optional<at::Tensor>& tensor) {
    if (tensor.has_value()) {
      (*this)(*tensor);
    }
  }
  bool short_circuit() {
    return out;
  }
};

template <typename... Args>
inline bool compute_requires_grad(Args&&... args) {
  if (!GradMode::is_enabled()) {
    return false;
  }
  return ComputeRequiresGrad().apply(std::forward<Args>(args)...).out;
}

inline void set_history(
    const at::Tensor& variable,
    const std::shared_ptr<Node>& grad_fn) {
  TORCH_CHECK(grad_fn != nullptr);
  if (variable.defined()) {
    // If the codegen triggers this, you most likely want to add your newly
    // added function to the DONT_REQUIRE_DERIVATIVE list in
    // tools/autograd/gen_variable_type.py
    TORCH_CHECK(
        isDifferentiableType(variable.scalar_type()),
        "Autograd not support dtype: ",
        variable.scalar_type());
    auto output_nr = grad_fn->add_input_metadata(variable);
    impl::set_gradient_edge(variable, {grad_fn, output_nr});
  } else {
    grad_fn->add_input_metadata(Node::undefined_input());
  }
}

inline void set_history(
    const std::vector<Variable>& variables,
    const std::shared_ptr<Node>& grad_fn) {
  for (auto& variable : variables) {
    set_history(variable, grad_fn);
  }
}

inline bool isFwGradDefined(const std::optional<at::Tensor>& t) {
  return t.has_value() && t->defined() && t->_fw_grad(/*level */ 0).defined();
}

inline bool isFwGradDefinedTensorList(const at::ITensorListRef& variables) {
  bool ret = false;
  for (auto& variable : variables) {
    ret |= isFwGradDefined(variable);
  }
  return ret;
}

inline bool isFwGradDefinedTensorList(
    const c10::List<std::optional<at::Tensor>>& li) {
  bool ret = false;
  for (auto i : c10::irange(li.size())) {
    auto t = li.get(i);
    ret |= isFwGradDefined(t);
  }
  return ret;
}

} // namespace torch::autograd
